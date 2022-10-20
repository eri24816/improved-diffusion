import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np

from .nn_utils import SkipConnection

from .vdiff import EinopsToAndFrom, RelAttention # video diffusion stuffs

from typing import Optional, Union, Callable, List, Tuple

class TransformerEncoderLayer(nn.Module):
    '''
    Modified version of the original TransformerEncoderLayer. Inserted a temporal attention block between the self-attention block and the feed forward block.
    input: [b f l d]
    output: [b f l d]
    '''
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,**factory_kwargs)
        self.self_attn = EinopsToAndFrom('b f l d', '(b f) l d',lambda x,*args,**kwargs: self.attn(x,x,x,*args,**kwargs)[0])
        self.temporal_attn = EinopsToAndFrom('b f l d', 'b l f d', RelAttention(d_model, heads = nhead)) #* no rotary embedding yet
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3= nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._temporal_attn_block(self.norm2(x))
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._temporal_attn_block(x))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)
        return self.dropout1(x)

    
    def _temporal_attn_block(self, x: Tensor) -> Tensor:
        x = self.temporal_attn(x)
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class FFTransformer(nn.Module):
    def __init__(self,d,n_blocks = 4, n_heads = 8, d_cond = 0, learn_sigma = False, Nonlinearity = nn.SiLU, frame_size = 1, init_out = None) -> None:
        super().__init__()

        self.init_out = init_out
        
        # time embbeding
        self.d_time_emb = 16
        self.time_emb = nn.Sequential(nn.Linear(1, self.d_time_emb),Nonlinearity())
 
        # positional embedding
        self.d_pos_emb = 11#+32
        pos = np.arange(0,frame_size)
        pos_emb = [(pos//(2**i))%2 for i in range(11)]#+[pos%32 == i for i in range(32)]
        pos_emb = np.stack(pos_emb).T

        #self.register_buffer('pos_emb',pos_emb.clone().contiguous())
        self.register_buffer('pos_emb',torch.tensor(pos_emb,dtype=torch.float).clone())

        # build model blocks
        self.frame_size = frame_size

        self.in_block = nn.Sequential(# [B,L,P] -> [B,L,D]
            nn.Linear(88+self.d_time_emb+self.d_pos_emb+d_cond,d),
            SkipConnection(
                Nonlinearity(),
                nn.Linear(d,d),
            )
        )

        self.transformer = nn.Sequential(*[TransformerEncoderLayer(d_model=d, nhead=n_heads,batch_first=True) for _ in range(n_blocks)])

        self.out_block = nn.Sequential(# [B,L,D] -> [B,L,P]
            nn.Linear(d,88)
        )

        if self.init_out is not None:
            # initialize out block to zero
            torch.nn.init.zeros_(self.out_block[0].weight)
            torch.nn.init.zeros_(self.out_block[0].bias)

        self.learn_sigma=learn_sigma
        if learn_sigma:
            self.out_sigma_block = nn.Sequential(# [B,L,D] -> [B,L,P]
                nn.Linear(d,88)
            )
    
    def forward(self,x,t,condition:Optional[torch.Tensor] = None):
        # x : [b, l=n_bar*32, p=88]
        # t : [b]

        B = x.shape[0]
        L = x.shape[1]
        frame_size = self.frame_size
        num_frames = L//frame_size

        t_emb = self.time_emb(t.view(-1,1)) # [B, 16]
        t_emb = t_emb.view(B,1,-1).expand(B,L,-1) # [B, L, 16]

        pos_emb = self.pos_emb[:frame_size].repeat([num_frames,1]) # type: ignore # [L, d_pos_emb]
        pos_emb = pos_emb.view(1,L,self.d_pos_emb).expand(B,L,self.d_pos_emb) # [B, L, d_pos_emb]

        if condition != None: # [B, L, d_latent]
            x = torch.cat([x,t_emb,pos_emb,condition],dim = -1) # [B, L, D]
        else:
            x = torch.cat([x,t_emb,pos_emb],dim = -1) # [B, L, D]

        x = x.view(B,num_frames,frame_size,-1) # [B, l, f, D] #TODO: use rearrange

        x = self.in_block(x)
        x = self.transformer(x)
        mu = self.out_block(x)
        mu = mu.view(B,L,88) # squeeze num_frames and frame_size #TODO: use rearrange

        if self.init_out is not None:
            mu = mu + self.init_out

        if self.learn_sigma:
            sigma = self.out_sigma_block(x)
            sigma = sigma.view(B,L,88) #TODO: use rearrange
            out = torch.cat([mu,sigma],dim=1) # that's the time dim, but the diffusion model code requires to cat on dim 1
        else:
            out = mu
        
        return out

    @classmethod
    def test(cls):
        instance = cls(16)
        x = torch.randn(2,32,88)
        t = torch.randn(2)
        out = instance(x,t)
        assert out.shape == x.shape
        return out
