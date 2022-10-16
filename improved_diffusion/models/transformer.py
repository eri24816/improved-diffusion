import torch
from torch import nn
import numpy as np
from nn_utils import SkipConnection
from typing import Optional

class FFTransformer(nn.Module):
    def __init__(self,d,n_blocks = 4, n_heads = 8, d_cond = 0, learn_sigma = False, Nonlinearity = nn.SiLU) -> None:
        super().__init__()
        
        # time embbeding
        self.d_time_emb = 16
        self.time_emb = nn.Sequential(nn.Linear(1, self.d_time_emb),Nonlinearity())
 
        # positional embedding
        self.d_pos_emb = 11#+32
        pos = np.arange(0,4096)
        pos_emb = [(pos//(2**i))%2 for i in range(11)]#+[pos%32 == i for i in range(32)]
        pos_emb = np.stack(pos_emb)
        pos_emb = torch.tensor(pos_emb).transpose(0,1)
        self.register_buffer('pos_emb',pos_emb)

        # build model blocks
        self.in_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(88+self.d_time_emb+self.d_pos_emb+d_cond,d),
            SkipConnection(
                Nonlinearity(),
                nn.Linear(d,d),
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=n_blocks)

        self.out_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(d,88)
        )

        self.learn_sigma=learn_sigma
        if learn_sigma:
            self.out_sigma_block = nn.Sequential(# [B,L,P,D]
                nn.Linear(d,88)
            )

        #print(self)
        #print(f'param count: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1_000_000}M')
            
    
    def forward(self,x,t,condition:Optional[torch.Tensor] = None):
        # x : [b, l=n_bar*32, p=88]
        # t : [b]
        B = x.shape[0]
        L = x.shape[1]

        t_emb = self.time_emb(t.view(-1,1)) # [B, 16]
        t_emb = t_emb.view(B,1,-1).expand(B,L,-1) # [B, L, 16]

        pos_emb = self.pos_emb[:L] # [L, 43]
        pos_emb = pos_emb.view(1,L,-1).expand(B,L,-1) # [B, L, 43]

        if condition != None: # [B, L, d_latent]
            x = torch.cat([x,t_emb,pos_emb,condition],dim = -1) # [B, L, D]
        else:
            x = torch.cat([x,t_emb,pos_emb],dim = -1) # [B, L, D]

        x = self.in_block(x)

        x = self.transformer(x)

        mu = self.out_block(x)
        
        if self.learn_sigma:
            sigma = self.out_sigma_block(x)
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