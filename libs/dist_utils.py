from typing import List
import os
import torch
import torch.distributed as dist
import logging

def barrier():
    if dist.is_initialized():
        dist.barrier()
    else:
        pass


def broadcast(data, src):
    if dist.is_initialized():
        dist.broadcast(data, src)
    else:
        pass


def all_gather(data: List, src):
    if dist.is_initialized():
        dist.all_gather(data, src)
    else:
        data[0] = src


def all_gather_object(data: List, src):
    if dist.is_initialized():
        dist.all_gather_object(data, src)
    else:
        data[0] = src


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
    

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True

def setup_logging(log_dir='logs', log_file_name='training.log'):
    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        log_format = '%(asctime)s  %(filename)s : %(levelname)s  %(message)s'
        date_format  = '%Y-%m-%d %A %H:%M:%S'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            filename=os.path.join(log_dir, log_file_name),
            filemode='a'
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logging.getLogger().addHandler(console_handler)
    else:
        logging.getLogger().addHandler(logging.NullHandler())

