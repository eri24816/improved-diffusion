import torch
import torch.nn as nn
import numpy as np
from utils import music
from .nn_utils import Nonlinearity, SkipConnection, Permute

def gaussian_sample(mean,logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

def kl_divergence(mean,logvar):
    return -0.5 * (logvar - mean.pow(2) - torch.exp(logvar) + 1).sum(1)

class Encoder(nn.Module):
    def __init__(self,d,n_blocks = 4, n_heads = 8, out_d = 16) -> None:
        super().__init__()
        self.out_d = out_d
        # positional embedding
        self.d_pos_emb = 11#+32
        pos = np.arange(0,4096)
        pos_emb = [(pos//(2**i))%2 for i in range(11)]#+[pos%32 == i for i in range(32)]
        pos_emb = np.stack(pos_emb)
        pos_emb = torch.tensor(pos_emb).transpose(0,1)
        self.register_buffer('pos_emb',pos_emb)

        # build model blocks
        self.in_block = nn.Sequential(# [B,L,D]
            nn.Linear(88+self.d_pos_emb,d),
            SkipConnection(
                Nonlinearity(),
                nn.Linear(d,d),
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=n_blocks)

        self.out_block = nn.Sequential(# [B,L,D]
            nn.Linear(d,out_d*2),
            Permute((0,2,1)),   # [B,out_d*2,L]
            nn.Linear(32,1),    # [B,out_d*2,1]
            nn.Flatten(1)       # [B,out_d*2]
        )

    def forward(self,x,sample=False,return_kl=False):
        # x : [b, l=n_bar*32, p=88]
        # t : [b]
        B = x.shape[0]
        L = x.shape[1]

        pos_emb = self.pos_emb[:L] # [L, 43]
        pos_emb = pos_emb.view(1,L,-1).expand(B,L,-1) # [B, L, 43]

        x = torch.cat([x,pos_emb],dim = -1) # [B, L, D]

        x = self.in_block(x)
        x = self.transformer(x) # [B, L, D]
        x = self.out_block(x) # [B, out_d*2]
        
        mean, logvar = x[:,:self.out_d], x[:,self.out_d:] # [B, out_d], [B, out_d]
        if sample:
            x = gaussian_sample(mean,logvar)
        else:
            x = mean
        if return_kl:
            return x, kl_divergence(mean,logvar) if sample else 0
        else:
            return x

class CyclicalKlWeight():
    def __init__(self,max = 1,period = 10000):
        self.max = max
        self.n_steps = period
        
    def get(self,step):
        return min(1,(2*step/self.n_steps)%2)*self.max