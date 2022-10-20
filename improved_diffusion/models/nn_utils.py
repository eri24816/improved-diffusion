from torch import nn
import torch
import math
from einops import rearrange

Nonlinearity = nn.SiLU

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

class SkipConnection(nn.Sequential):
    def forward(self, input):
        return input + super().forward(input)

# copied from https://github.com/lucidrains/video-diffusion-pytorch/blob/main/video_diffusion_pytorch/video_diffusion_pytorch.py

class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

def count_parameters(model):
    print(f'param count of {type(model)}: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1_000_000}M')

class SpaceTemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.relative_position_bias = RelativePositionBias(heads = heads)

    def forward(self, x, device):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        kv = self.to_kv(x)
        k, v = rearrange(kv, 'b n (h k v) -> b h n k v', h = h)

        q, k, v = map(lambda t: rearrange(t, 'b h n d -> b h n () d'), (q, k, v))

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        relative_position_bias = self.relative_position_bias(n, device)
        dots = dots + relative_position_bias

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n () d -> b n h d')
        out = self.to_out(out)

        return out