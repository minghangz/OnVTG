from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import os
import shutil
import time
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .dataset import make_dataset, make_dataloader
from .dist_utils import get_rank, get_world_size, barrier, all_gather, is_main_process
from .modeling import (
    OnVTG, PtGenerator,
    sigmoid_focal_loss, ctr_giou_loss, ctr_diou_loss,
    make_optimizer, make_scheduler
)
from .train_utils import AverageMeter, fix_random_seed, time_str
from .evaluator import Evaluator

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt['seed'])

        # build model and EMA
        self.model = OnVTG(opt['model']).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(
            opt['train']['data'], num_epochs=self.num_epochs, is_training=True
        )
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.warmup_epochs = opt['train']['scheduler']['warmup_epochs'] = opt['train']['warmup_epochs']
        opt['train']['scheduler']['epochs'] = opt['train']['epochs']
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.tb_writer = self.loss_meters = self.timer = None

        # load model weights and training states
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()])
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['vid_net']['max_seq_len']
        self.max_text_len = opt['model']['text_net']['max_seq_len']
        self.vid_stride = opt['model']['vid_net']['stride']
        self.num_fpn_levels = len(opt['model']['vid_net']['memory_size'])
        self.input_vid_len = self.max_vid_len * self.vid_stride
        self.pt_gen = PtGenerator(max_seq_len=opt['train']['data']['long_term_window_size'], num_fpn_levels=self.num_fpn_levels).cuda()

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.future_loss_norm = opt['train']['future_loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.future_loss_weight = opt['train'].get('future_loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

        self.eval = opt['train']['eval']
        if self.eval:
            self.eval_epoch_interval = opt['train']['eval_epoch_interval']
            self.evaluator = Evaluator(opt)
            self.best_score = 0.0

    def run(self):
        logger.info("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            pbar = tqdm(self.dataloader) if is_main_process() else self.dataloader
            for data_list in pbar:
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
            is_best = False
            if self.eval and self.epoch % self.eval_epoch_interval == 0 and self.epoch >= self.warmup_epochs:
                self.evaluator.set_model(self.model_ema)
                stop_score = self.evaluator.run()
                logger.info("Eval: "+str(stop_score))
                if stop_score > self.best_score:
                    logger.info("Best score updated: " + str(stop_score))
                    self.best_score = stop_score
                    is_best = True

            self.checkpoint(is_best=is_best)
            barrier()
        logger.info("Training completed.")

    
    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = future_cls_loss = future_reg_loss = norm = future_norm = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            future_cls_loss += loss_dict['future_cls']
            future_reg_loss += loss_dict['future_reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']
            future_norm += loss_dict['future_norm']

        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        all_norms = [torch.zeros_like(future_norm) for _ in range(get_world_size())]
        all_gather(all_norms, future_norm)
        self.future_loss_norm = (
            self.loss_norm_momentum * self.future_loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {'cls': cls_loss, 'reg': reg_loss, 'future_cls': future_cls_loss, 'future_reg': future_reg_loss, 'total': total_loss}

    def _microbatch_forward_backward(self, data_list, is_last=False):
        self.model.train()
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits, fpn_offsets, fpn_masks, future_fpn_logits, future_fpn_offsets = \
                self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_offsets, fpn_masks, future_fpn_logits, future_fpn_offsets = \
                    self.model(vid, vid_masks, text, text_masks, text_size)
        
        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True).flatten(0, 1)

        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        future_fpn_masks = fpn_masks[0].flatten(0, 1)
        fpn_logits = torch.cat(fpn_logits, dim=2).flatten(0, 1)
        fpn_offsets = torch.cat(fpn_offsets, dim=2).flatten(0, 1)
        fpn_masks = torch.cat(fpn_masks, dim=2).flatten(0, 1)
        points = torch.cat(fpn_points)
        if future_fpn_logits is not None:
            future_fpn_logits = torch.cat(future_fpn_logits, dim=2).flatten(0, 1)
        if future_fpn_offsets is not None:
            future_fpn_offsets = torch.cat(future_fpn_offsets, dim=2).flatten(0, 1)

        # annotate points
        gt_labels, gt_offsets, future_labels, future_offsets = self._annotate_points(points, targets)
        future_labels = future_labels[:, :future_fpn_masks.size(1)]
        future_offsets = future_offsets[:, :future_fpn_masks.size(1)]
        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()
        pos_future_masks = torch.logical_and(future_labels, future_fpn_masks)
        future_norm = pos_future_masks.sum()

        ## (2) classification loss on valid points
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()

        if future_fpn_logits is not None:
            future_cls_loss = self._calc_focal_loss(
                logits=future_fpn_logits[future_fpn_masks], labels=future_labels[future_fpn_masks]
            ) / self.future_loss_norm * get_world_size()
        else:
            future_cls_loss = torch.tensor(0.0)
        
        if future_fpn_offsets is not None:
            future_reg_loss = F.l1_loss(
                future_fpn_offsets[pos_future_masks].squeeze(-1), future_offsets[pos_future_masks], reduction='sum'
            ) / self.future_loss_norm * get_world_size()
        else:
            future_reg_loss = torch.tensor(0.0)

        total_loss = cls_loss + self.loss_weight * reg_loss + self.future_loss_weight * (future_reg_loss + future_cls_loss)
        total_loss.backward()
        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'future_cls': future_cls_loss.detach(),
            'future_reg': future_reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
            'future_norm': future_norm.detach(),
        }

    def _batchify_videos(self, vid_list):
        bs, nw = len(vid_list), len(vid_list[0])
        vid_list = [vid for window_list in vid_list for vid in window_list]
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        input_vid_len = max(vid_lens)
        min_vid_lens = 2 ** (self.num_fpn_levels - 1) * self.vid_stride
        input_vid_len = math.ceil(input_vid_len / min_vid_lens) * min_vid_lens
        vid = vid_list[0].new_full((bs * nw, vid_dim, input_vid_len), 0.)
        for idx in range(bs * nw):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(input_vid_len)[None] < vid_lens
        return vid.view(bs, nw, vid_dim, input_vid_len), vid_masks.view(bs, nw, input_vid_len)

    def _batchify_text(self, text_list):
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])

        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks

    def _batchify(self, vid_list, text_list):
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)

            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        return vid, vid_masks, text, text_masks, text_size

    def _annotate_points(self, points, targets):
        labels_list, offsets_list, future_labels_list, future_offsets_list = tuple(), tuple(), tuple(), tuple()
        for target in targets:
            labels, offsets, future_labels, future_offsets = self._annotate_points_per_video(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
            future_labels_list += (future_labels, )
            future_offsets_list += (future_offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        future_labels = torch.stack(future_labels_list)
        future_offsets = torch.stack(future_offsets_list)
        return labels, offsets, future_labels, future_offsets

    def _annotate_points_per_video(self, points, target):
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]
        pt2end = target[1] - points[:, 0]

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0] 
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        # future_offsets = target[0] - (points[:, 0] + points[:, 0])
        future_labels = torch.logical_and(pt2start < 4, pt2start > -4)

        return labels, offsets, future_labels, pt2start

    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu')
        state_ckpt = torch.load(state_path, map_location='cpu')
        self.model.load_state_dict(model_ckpt['model'])
        self.model_ema.load_state_dict(model_ckpt['model_ema'])
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        logger.info(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self, is_best=False):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        logger.info(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        Path(model_dir).mkdir(exist_ok=True)
        Path(state_dir).mkdir(exist_ok=True)
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        if self.epoch in self.checkpoint_epochs:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
            )
        if is_best:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, 'best.pth')
            )
            shutil.copyfile(
                os.path.join(state_dir, 'last.pth'),
                os.path.join(state_dir, 'best.pth')
            )

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        logger.info(log_str)
        self.tb_writer.flush()
