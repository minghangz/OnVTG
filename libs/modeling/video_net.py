from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .blocks import (
    sinusoid_encoding, MaskedConv1D, LayerNorm, masked_max_pool1d, LayerScale, LayerNorm, FFN
)

backbones = dict()
def register_video_net(name):
    def decorator(module):
        backbones[name] = module
        return module
    return decorator


class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask
    NOTE: This implementation supports
        - global and local self-attention
        - (global) cross-attention

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        q_dim=None,         # query dimension
        kv_dim=None,        # key / value dimension
        out_dim=None,       # output dimension
        n_heads=4,          # number of attention heads
        window_size=0,      # local attention window size (0 for global attention)
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
    ):
        super(MaskedMHA, self).__init__()

        assert embd_dim % n_heads == 0
        self.embd_dim = embd_dim

        if q_dim is None:
            q_dim = embd_dim
        if kv_dim is None:
            kv_dim = embd_dim
        if out_dim is None:
            out_dim = q_dim

        self.n_heads = n_heads
        self.n_channels = embd_dim // n_heads
        self.scale = 1.0 / np.sqrt(np.sqrt(self.n_channels))
        self.out_dim = out_dim

        self.query = nn.Conv1d(q_dim, embd_dim, 1)
        self.key = nn.Conv1d(kv_dim, embd_dim, 1)
        self.value = nn.Conv1d(kv_dim, embd_dim, 1)
        self.proj = nn.Conv1d(embd_dim, out_dim, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # local attention window size
        assert window_size == 0 or window_size % 2 == 1
        self.window_size = window_size
        self.stride = window_size // 2

        # masks for local attention (left / right paddings)
        self.l_mask = self.r_mask = None

    def forward(self, x, mask, history=None, history_mask=None):
        bs, c = x.size(0), self.embd_dim
        h, d, w = self.n_heads, self.n_channels, self.window_size

        q = self.query(x)
        k, history_k = self.key(x), self.key(history) if history is not None else None
        v, history_v = self.value(x), self.value(history) if history is not None else None

        q = q.view(bs, h, d, -1).transpose(2, 3).contiguous()
        k = k.view(bs, h, d, -1)
        v = v.view(bs, h, d, -1).transpose(2, 3).contiguous()
        if history is not None:
            history_k = history_k.view(bs, h, d, -1)
            history_v = history_v.view(bs, h, d, -1).transpose(2, 3).contiguous()

            k = torch.cat([k, history_k], dim=-1)
            v = torch.cat([v, history_v], dim=2)
            mask = torch.cat([mask, history_mask], dim=-1)

        attn = (q * self.scale) @ (k * self.scale)      # (bs, h, t1, t2)
        attn = attn.masked_fill(
            mask=torch.logical_not(mask[:, :, None, :]),
            value=-1e9,
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        q = attn @ v                                    # (bs, h, t1, d)

        q = q.transpose(2, 3).contiguous().view(bs, c, -1)            # (bs, c, t1)
        out = self.proj_drop(self.proj(q))

        return out
    

class OnlineTransformerEncoder(nn.Module):
    """
    Transformer Encoder.
    (optional depth-wise conv -> local self-attn -> FFN)
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        n_heads=4,          # number of attention heads
        window_size=0,      # MHA window size (0 for global attention)
        expansion=4,        # expansion factor for FFN
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
    ):
        super(OnlineTransformerEncoder, self).__init__()

        # self-attention
        self.attn = MaskedMHA(
            embd_dim, n_heads=n_heads, window_size=window_size,
            attn_pdrop=attn_pdrop, proj_pdrop=proj_pdrop,
        )
        self.ln_attn = LayerNorm(embd_dim)
        self.drop_path_attn = LayerScale(embd_dim, path_pdrop)

        # FFN
        self.ffn = FFN(embd_dim, expansion, proj_pdrop)
        self.ln_ffn = LayerNorm(embd_dim)
        self.drop_path_ffn = LayerScale(embd_dim, path_pdrop)

    def forward(self, x, mask, history=None, history_mask=None):
        if mask is None:
            mask = torch.ones_like(x[:, :1], dtype=torch.bool)
        x = x * mask.to(x.dtype)

        if history is None:
            h = self.attn(self.ln_attn(x), mask) * mask.to(x.dtype)
        else:
            history = history * history_mask.to(x.dtype)
            h = self.attn(self.ln_attn(x), mask, self.ln_attn(history), history_mask) * mask.to(x.dtype)
        x = x * mask.to(x.dtype) + self.drop_path_attn(h)

        # FFN
        h = self.ffn(self.ln_ffn(x)) * mask.to(x.dtype)
        x = x + self.drop_path_ffn(h)

        return x, mask


class EventMemory:
    def __init__(self, n_scales, memory_size, threshold=0.95):
        self.n_scales = n_scales
        self.memory_size = memory_size
        self.history = []
        self.history_mask = []
        self.threshold = threshold

    def read(self, scale):
        if len(self.history) <= scale:
            return None, None
        return self.history[scale], self.history_mask[scale]

    def update(self, scale, x, mask):
        if len(self.history) <= scale:
            self.history.append(x)
            self.history_mask.append(mask)
        else:
            self.history[scale] = torch.cat([self.history[scale], x], dim=-1)
            self.history_mask[scale] = torch.cat([self.history_mask[scale], mask], dim=-1)
        
        if self.history[scale].size(-1) > self.memory_size[scale]:
            self.history[scale] = self.history[scale][..., 1:]
            self.history_mask[scale] = self.history_mask[scale][..., 1:]
    
    def adaptive_merge(self, T):
        B, dim, length = T.shape
        device = T.device

        t1 = T[:, :, :-1]
        t2 = T[:, :, 1:]
        similarities = F.cosine_similarity(t1, t2, dim=1)
        max_sim_values, max_sim_indices = torch.max(similarities, dim=1)
        should_merge_mask = max_sim_values > self.threshold
        
        pooled_vectors = (t1 + t2) / 2.0
        source_for_merge = torch.cat([T, pooled_vectors], dim=2)
        k = max_sim_indices.unsqueeze(1) # (B, 1)
        base_indices = torch.arange(length - 1, device=device).expand(B, -1) #  (B, len-1)


        indices_lt_k = base_indices
        indices_eq_k = base_indices + length
        indices_gt_k = base_indices + 1
        merge_gather_indices = torch.where(
            base_indices < k, 
            indices_lt_k, 
            torch.where(base_indices == k, indices_eq_k, indices_gt_k)
        )
        merge_gather_indices = merge_gather_indices.unsqueeze(1).expand(B, dim, length - 1)
        merged_result = torch.gather(source_for_merge, 2, merge_gather_indices)

        discarded_result = T[:, :, 1:]

        final_mask = should_merge_mask.view(B, 1, 1).expand_as(merged_result)
        final_output = torch.where(final_mask, merged_result, discarded_result)

        return final_output
        
    def clear(self):
        self.history.clear()
        self.history_mask.clear()


@register_video_net('online_transformer')
class OnlineVideoTransformer(nn.Module):
    def __init__(
        self,
        in_dim,
        embd_dim,
        max_seq_len,
        n_heads,
        stride=1,
        n_convs=2,
        n_encoder_layers=8,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        use_abs_pe=False,
        use_ref_pe=False,
        short_window_size=8,
        memory_size=[8, 8, 8, 8, 8, 8, 8, 8],
        **kargs,
    ):
        super().__init__()

        assert stride & (stride - 1) == 0
        assert n_convs >= int(math.log2(stride))
        self.max_seq_len = max_seq_len
        self.stride = stride

        # embedding projection
        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1, 1, 0)

        # embedding convs

        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()
        for _ in range(n_convs):
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=5 if stride > 1 else 3,
                    stride=2 if stride > 1 else 1,
                    padding=2 if stride > 1 else 1,
                    bias=False
                )
            )
            self.embd_norms.append(LayerNorm(embd_dim))
            stride = max(stride // 2, 1)

        # position encoding (c, t)
        if use_abs_pe:
            self.pe_type = 'abs'
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        elif use_ref_pe:
            self.pe_type = 'ref'
            self.max_pe = 256
            self.pe = nn.Parameter(torch.randn(embd_dim, self.max_pe) / embd_dim ** 0.5)
        else:
            self.pe = None

        # branch transformers (for FPN)
        self.branch = nn.ModuleList()
        for idx in range(n_encoder_layers):
            self.branch.append(
                OnlineTransformerEncoder(
                    embd_dim,
                    n_heads=n_heads,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop
                )
            )
        self.short_window_size = short_window_size
        self.memory = EventMemory(n_encoder_layers, memory_size)
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def apply_pe(self, x, mask):
        if self.pe_type == 'ref':
            pe_all = []
            b, d, n = x.shape
            for i in range(b):
                t = mask[i].sum().item()
                pe = F.interpolate(self.pe[None], size=t, mode='nearest')
                if t < n:
                    pe = torch.cat([pe, torch.zeros((1, d, n - t), dtype=pe.dtype, device=pe.device)], dim=-1)
                pe_all.append(pe)
            pe_all = torch.cat(pe_all, dim=0)
            x = x + pe_all
        else:
            _, _, t = x.size()
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)

        return x

    def forward(self, x, mask):
        bs, nw, _, vlen = x.shape
        x = x.view(bs * nw, -1, vlen)
        mask = mask.view(bs * nw, vlen)

        self.memory.clear()
        if mask.ndim == 2:
            mask = mask.unsqueeze(1) 

        # embedding projection
        x, _ = self.embd_fc(x, mask)

        # embedding convs
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        # position encoding
        if self.pe is not None:
            x = self.apply_pe(x, mask)

        fpn, fpn_masks = [], []
        for i in range(0, x.size(-1), self.short_window_size):
            current = x[..., i: i+self.short_window_size]
            current_mask = mask[..., i: i+self.short_window_size]
            for scale in range(len(self.branch)):
                history, history_mask = self.memory.read(scale)
                current, current_mask = self.branch[scale](current, current_mask, history, history_mask)
                self.memory.update(scale, current, current_mask)
                if len(fpn) <= scale:
                    fpn.append(current.view(bs, nw, current.size(1), -1))
                    fpn_masks.append(current_mask.view(bs, nw, 1, -1))
                else:
                    fpn[scale] = torch.cat([fpn[scale], current.view(bs, nw, current.size(1), -1)], dim=-1)
                    fpn_masks[scale] = torch.cat([fpn_masks[scale], current_mask.view(bs, nw, 1, -1)], dim=-1)
                if scale + 1 == len(self.branch):
                    break
                if (i + self.short_window_size) % (2 ** (scale + 1)) != 0:
                    break
                if current.size(-1) % 2 != 0:
                    current = torch.cat([history[..., -1:], current], dim=-1)
                    current_mask = torch.cat([history_mask[..., -1:], current_mask], dim=-1)
                current, current_mask = masked_max_pool1d(current, current_mask, kernel_size=2, stride=2)
        
        return tuple(fpn), tuple(fpn_masks)
    

def make_video_net(opt):
    opt = deepcopy(opt)
    return backbones[opt.pop('name')](**opt)