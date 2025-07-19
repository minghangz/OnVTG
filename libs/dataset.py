from collections import OrderedDict
from functools import partial
import json
import h5py
import os
import random
from random import shuffle
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchtext
from torchtext.data import get_tokenizer


class OnVTGDataset(Dataset):

    def __init__(
        self,
        split,
        anno_file,
        vid_feat_dir,
        text_feat_dir,
        use_tokenizer,
        is_training,

        long_term_window_size,
        max_num_window,
        max_text_len,
        max_num_text,
        clip_size,
        clip_stride,
        downsample_rate,
        
        normalize_vid=False,
        normalize_text=False,
        drop_text=False,

        crop_ratio=(0.9, 1.0),
        trunc_thresh=0.5,

        num_epochs=1,
    ):
        super(OnVTGDataset, self).__init__()

        assert os.path.exists(anno_file)
        if not use_tokenizer:
            assert text_feat_dir is not None, (
                "text features must be given if tokenizer is not specified"
            )
        assert isinstance(downsample_rate, int) and downsample_rate >= 1
        if crop_ratio is not None:
            assert isinstance(crop_ratio, (list, tuple))

        self.split = split
        self.is_training = is_training
        self.epoch = 0  # this must be updated upon starting a new epoch

        self.anno_file = anno_file
        self.vid_feat_dir = vid_feat_dir
        self.text_feat_dir = text_feat_dir
        if self.vid_feat_dir.endswith('.h5'):
            self.vid_h5 = None
        if self.text_feat_dir is not None and self.text_feat_dir.endswith('.h5'):
            self.text_h5 = None
        self.tokenizer = GloVeTokenizer() if use_tokenizer else None

        self.long_term_window_size = long_term_window_size
        self.max_num_window = max_num_window
        self.max_text_len = max_text_len
        self.max_num_text = max_num_text
        self.clip_size = clip_size
        self.clip_stride = clip_stride * downsample_rate
        self.downsample_rate = downsample_rate

        self.normalize_vid = normalize_vid
        self.normalize_text = normalize_text
        self.drop_text = drop_text

        self.crop_ratio = crop_ratio
        self.trunc_thresh = trunc_thresh

        self.vid_dict, self.text_dict = self._parse_annotations()
        self.num_epochs = num_epochs

        if self.is_training:
            self.data_list = self._build_train_samples()
        else:
            assert self.num_epochs == 1
            self.data_list = self._build_eval_samples()

    def _parse_annotations(self):
        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        vid_dict, text_dict = OrderedDict(), OrderedDict()
        for key, value in anno[self.split].items():
            fps, num_frames = float(value['fps']), int(value['num_frames'])
            if 'duration' in value:
                duration = float(value['duration'])
            else:
                duration = num_frames / fps
            num_clips = (value['num_clips'] + self.downsample_rate - 1) // self.downsample_rate

            text_ids, segments = tuple(), tuple()
            for s, pair in enumerate(value['annotations']):
                text = pair['sentence'].strip()
                text_id = pair.get('sentence_id', key + '_{:04d}'.format(s))

                start = max(float(pair['segment'][0]), 0)
                end = min(float(pair['segment'][1]), duration)
                seg_len = end - start
                if seg_len <= 0:
                    continue
                segment = (start, end)
                segments += (segment, )
                text_dict[text_id] = {
                    'text'      : text,
                    'segment'   : np.array(segment),
                    'text_idx'  : s,
                    'vid_id'    : key,
                }
                text_ids += (text_id, )
            
            if len(text_ids) == 0:
                continue

            vid_dict[key] = {
                'fps'       : fps,
                'num_frames': num_frames,
                'num_clips' : num_clips,
                'duration'  : duration,
                'text_ids'  : text_ids,
                'segments'  : np.array(segments),
            }

        return vid_dict, text_dict
    

    def _load_vid_feats(self, vid_id):
        if self.vid_feat_dir.endswith('.h5'):
            if self.vid_h5 is None:
                self.vid_h5 = h5py.File(self.vid_feat_dir)
            vid_feats = self.vid_h5[vid_id][:].astype(np.float32)
        else:
            vid_feat_file = os.path.join(self.vid_feat_dir, vid_id + '.npy')
            vid_feats = np.load(vid_feat_file).astype(np.float32)

        if self.downsample_rate > 1:
            vid_feats = vid_feats[::self.downsample_rate]

        vid_feats = vid_feats.transpose()                   # (c, t)
        vid_feats = torch.from_numpy(np.ascontiguousarray(vid_feats))

        # normalize features to unit length
        if self.normalize_vid:
            vid_feats = F.normalize(vid_feats, dim=0)
        return vid_feats

    def _truncate_vid_feats(
        self,
        feats,
        segments,
        offset,
        num_trials=5000
    ):
        vid_len = feats.size(1)
        max_vid_len = self.long_term_window_size

        if vid_len <= max_vid_len:
            if self.crop_ratio is None:
                return feats, 0, vid_len

            max_vid_len = random.randint(
                max(np.ceil(self.crop_ratio[0] * vid_len), 1),
                min(np.ceil(self.crop_ratio[1] * vid_len), vid_len)
            )
            if max_vid_len == vid_len:
                return feats, 0, vid_len

        # rough estimate on the range of valid chunks
        s0 = max(0, np.floor(segments[:, 0].min() - max_vid_len))
        s1 = min(vid_len - max_vid_len, np.ceil(segments[:, 1].max()))
        
        seg_lens = torch.clamp(segments[:, 1] - segments[:, 0], min=1e-5)

        for _ in range(num_trials):
            ws = random.randint(s0, s1) # window start
            we = ws + max_vid_len       # window end

            # check overlap with segments
            start = torch.clamp(segments[:, 0], min=ws - offset)
            end = torch.clamp(segments[:, 1], max=we + offset)
            overlap = torch.clamp(end - start, min=0)
            if torch.all(overlap / seg_lens > self.trunc_thresh):
                feats = feats[:, ws:we]
                return feats, ws, we

        feats = feats[:, ws:we]
        return feats, ws, we

    def _load_text_feats(self, text_id):
        if self.tokenizer is not None:
            text_feats = self.tokenizer(self.text_dict[text_id]['text'])
        elif self.text_feat_dir.endswith('.h5'):
            if self.text_h5 is None:
                self.text_h5 = h5py.File(self.text_feat_dir)
            text_feats = self.text_h5[text_id][:].astype(np.float32)
            text_feats = text_feats.transpose()     # (c, t)
            text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
        else:
            text_feat_file = os.path.join(self.text_feat_dir, text_id + '.npy')
            text_feats = np.load(text_feat_file).astype(np.float32)
            text_feats = text_feats.transpose()     # (c, t)
            text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
            
        if self.is_training:
            text_feats = text_feats[:, :self.max_text_len]

        # normalize text features to unit length
        if self.normalize_text:
            text_feats = F.normalize(text_feats, dim=0)

        return text_feats

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def _build_train_samples(self):
        samples = []
        for _ in range(self.num_epochs):
            for vid_id in self.vid_dict.keys():
                video_samples = self._group(vid_id)
                shuffle(video_samples)
                if self.drop_text:
                    samples.append((vid_id, video_samples[:1]))
                    continue
                for i in range(0, len(video_samples), self.max_num_window):
                    samples.append((vid_id, video_samples[i:i+self.max_num_window]))
                if len(video_samples) % self.max_num_window != 0 and len(video_samples) > self.max_num_window:
                    samples.append((vid_id, video_samples[-self.max_num_window:]))
        return tuple(samples)

    def _build_eval_samples(self):
        samples = []
        for vid_id, vid_dict in self.vid_dict.items():
            samples += [(vid_id, [tuple(range(len(vid_dict['segments'])))])]
        return tuple(samples)

    def _group(self, vid_id):
        vid_dict = self.vid_dict[vid_id]

        if vid_dict['num_clips'] <= self.long_term_window_size:
            win_len = vid_dict['num_clips']
            if self.crop_ratio is not None:
                win_len = max(np.ceil(self.crop_ratio[0] * win_len), 1)
        else:
            win_len = self.long_term_window_size
        win_len = (
            self.clip_stride * (win_len - 1) + self.clip_size
        ) / vid_dict['fps'] 

        sort_idx = np.argsort(vid_dict['segments'][:, 0])
        segments = vid_dict['segments'][sort_idx]
        mask = np.ones(len(segments), dtype=bool)

        samples = []
        while mask.sum() > 0:
            ptr = np.nonzero(mask)[0].min()

            ws, we = segments[ptr, 0], segments[ptr, 0] + win_len
            if segments[ptr, 1] - segments[ptr, 0] > win_len:
                idx = np.array([ptr])
            else:
                is_inside = (
                    (segments[:, 0] >= ws) & (segments[:, 1] <= we) & mask
                )
                idx = np.nonzero(is_inside)[0]
                if len(idx) > self.max_num_text:
                    idx = np.random.choice(idx, self.max_num_text, replace=False)
            sample = tuple(sort_idx[idx])
            samples += [sample]
            mask[idx] = 0
        return samples

    def __len__(self):
        return len(self.data_list) // self.num_epochs

    def __getitem__(self, idx):
        vid_id, seg_idx = self.data_list[self.epoch * len(self) + idx]
        vid_dict = self.vid_dict[vid_id]

        vid_feats = self._load_vid_feats(vid_id)

        clip_size, clip_stride = self.clip_size, self.clip_stride
        clip_offset = 0.5 * clip_size / clip_stride

        segments = []
        for sid in seg_idx:
            segs = np.clip(vid_dict['segments'][np.array(sid)] * vid_dict['fps'], a_min=0, a_max=vid_dict['num_frames']) / clip_stride - clip_offset
            segs = torch.from_numpy(
                np.ascontiguousarray(segs.astype(np.float32))
            )
            segments.append(segs)
        segments_cat = torch.cat(segments, dim=0)

        all_window_feats = tuple()
        all_segments = tuple()
        for segs in segments:
            if self.is_training:
                window_feats, ws, we = self._truncate_vid_feats(
                    vid_feats, segs, clip_offset
                )
                all_window_feats += (window_feats, )
                all_segments += (segments_cat - ws, )
            else:
                all_segments = (segments_cat, )
                all_window_feats = (vid_feats, )
        all_segments = torch.stack(all_segments, dim=1)

        text_feats_list = tuple()
        for sid in seg_idx:
            for idx in sid:
                text_feats = self._load_text_feats(vid_dict['text_ids'][idx])
                text_feats_list += (text_feats, )


        return {
            'fps'        : vid_dict['fps'],
            'num_frames' : vid_dict['num_frames'],
            'duration'   : vid_dict['duration'],
            'segment'    : vid_dict['segments'],
            'clip_size'  : clip_size,
            'clip_stride': clip_stride,
            'target'     : all_segments,
            'vid'        : all_window_feats,
            'text'       : text_feats_list,
        }

class GloVeTokenizer:

    def __init__(self, name='6B'):

        self.vocab = torchtext.vocab.GloVe(name=name, cache='/network_space/storage43/zhengmh/Dataset/glove/')
        self.tokenizer = get_tokenizer("basic_english")

    def __call__(self, text, max_len=None):
        words = self.tokenizer(text)
        feats = self.vocab.get_vecs_by_tokens(words, lower_case_backup=True)
        if max_len is not None:
            feats = feats[:max_len]
        feats = feats.transpose(0, 1)

        return feats

def make_dataset(opt, num_epochs=1, is_training=True):
    opt = deepcopy(opt)
    return OnVTGDataset(is_training=is_training, num_epochs=num_epochs,  **opt)

def trivial_batch_collator(batch):
    return batch


def worker_init_reset_seed(worker_id, num_workers, rank):
    seed = torch.initial_seed() % 2 ** 31
    worker_seed = num_workers * rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)

    
def make_dataloader(
    dataset,
    generator,
    batch_size,
    num_workers,
    is_training,
    world_size=1,
    rank=0,
):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=is_training, drop_last=is_training)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=partial(worker_init_reset_seed, num_workers, rank),
        sampler=sampler,
        shuffle=(sampler is None and is_training),
        drop_last=is_training,
        generator=generator,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader, sampler