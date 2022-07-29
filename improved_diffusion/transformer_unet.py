from turtle import forward
from regex import D
import torch
import torch.nn as nn
import numpy as np
from utils import music

Nonlinearity = nn.SiLU

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

class SkipConnection(nn.Sequential):
    def forward(self, input):
        return input + super().forward(input)

# https://github.com/tunz/transformer-pytorch/blob/e7266679f0b32fd99135ea617213f986ceede056/model/transformer.py
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_size, dropout_rate = 0):
        super().__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_qkv = nn.Linear(hidden_size, head_size * att_size * 3, bias=False)
        initialize_weight(self.linear_qkv)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, x):
        # x : [B,L,D]
        d_k = self.att_size
        d_v = self.att_size
        batch_size = x.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        qkv = self.linear_qkv(x).view(batch_size, -1, 3, self.head_size, d_k).transpose(1, 3) # [b, h, (qkv), len, d]

        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2] # [b, h, len, d]

        k = k.transpose(2, 3)  # [b, h, d, len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, len, len]
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads = 8) -> None:
        super().__init__()
        self.atten_norm = nn.LayerNorm(d)
        #self.attention = SelfAttention(d, n_heads)
        self.attention = nn.MultiheadAttention(d, n_heads)
        self.linear_norm = nn.LayerNorm(d)
        self.positionwise_linear = nn.Sequential(
            nn.Linear(d, d),
            Nonlinearity(),
            nn.Linear(d,d),
        )

    def forward(self,x):
        # [B, L, D]
        t = self.atten_norm(x)
        x = x + self.attention(t,t,t,need_weights=False)[0]
        #x = x + self.positionwise_linear(self.linear_norm(x))
        x = self.positionwise_linear(self.linear_norm(x))
        return x

class PitchAwareBlock(nn.Module):
    def __init__(self,in_channels,out_channels,d) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels,d,5,padding=2),
            Nonlinearity(),
            nn.Conv1d(d,d,5,padding=2),
        )
        self.attn = SelfAttention(d,8)
        self.linear = nn.Linear(in_channels+d,out_channels)
        self.D = d
    def forward(self,x):
        # x : [B,L,P,D]
        B, L, P, IN_D = x.shape
        D = self.D
        
        y = x
        x = x.permute(0,1,3,2).reshape(B*L,IN_D,P) # [B*L,D,P]
        x = self.conv(x) # [B*L,D,P]
        x = x.view(B,L,D,P).permute(0,1,3,2) # [B,L,P,D]
        
        x = x.permute(0,2,1,3).reshape(B*P,L,D) # [B*P,L,D]
        x = self.attn(x) # [B*P,L,D]
        x = x.view(B,P,L,D).permute(0,2,1,3) # [B,L,P,D]

        x = torch.cat([x,y],dim = -1)

        x = self.linear(x)

        return x


class PitchAwareTransformerUnet(nn.Module):
    def __init__(self,d,n_blocks = 2, n_heads = 8, learn_sigma = False) -> None:
        super().__init__()

        # time embbeding
        self.d_time_emb = 16
        self.time_emb = nn.Sequential(nn.Linear(1, self.d_time_emb),Nonlinearity())

        # positional embedding
        self.d_pos_emb = 11+32
        pos = np.arange(0,4096)
        pos_emb = [(pos//(2**i))%2 for i in range(11)]+[pos%32 == i for i in range(32)]
        pos_emb = np.stack(pos_emb)
        pos_emb = torch.tensor(pos_emb).transpose(0,1)
        self.register_buffer('pos_emb',pos_emb)

        # build model blocks

        d_p_block_inner = 8
        d_p_block_hidden = 16
        self.in_p_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(1+self.d_time_emb+self.d_pos_emb,d_p_block_hidden),
            Nonlinearity(),
            PitchAwareBlock(d_p_block_hidden,d_p_block_hidden,d_p_block_hidden),
            PitchAwareBlock(d_p_block_hidden,d_p_block_inner,d_p_block_hidden)
        )

        d_transformer_in = d_p_block_inner*52 + 88 + self.d_time_emb + self.d_pos_emb
        d_transformer_out = d_p_block_inner*52

        self.interlayer_down = nn.Linear(d_transformer_in,d)

        self.down1 = nn.Sequential(
            *[TransformerBlock(d, n_heads) for i in range(n_blocks)]
        )
        self.up1 = nn.Sequential(
            *[TransformerBlock(d, n_heads) for i in range(n_blocks)]
        )

        self.interlayer_up = nn.Linear(d,d_transformer_out)
        
        self.out_p_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(d_p_block_inner*2,d_p_block_hidden),
            Nonlinearity(),
            PitchAwareBlock(d_p_block_hidden,d_p_block_hidden,d_p_block_hidden),
            PitchAwareBlock(d_p_block_hidden,d_p_block_hidden,d_p_block_hidden)
        )

        self.out_p_block_88 = nn.Sequential(# [B,L,P,D]
            PitchAwareBlock(d_p_block_hidden,d_p_block_hidden,d_p_block_hidden),
            PitchAwareBlock(d_p_block_hidden,d_p_block_hidden,d_p_block_hidden),
            nn.Linear(d_p_block_hidden,1),
        )

        print(self)
        print(f'param count:{sum(p.numel() for p in self.parameters() if p.requires_grad)}')
            
    
    def forward(self,x,t):
        # x : [b, l=n_bar*32, p=88]
        # t : [b]
        B = x.shape[0]
        L = x.shape[1]

        t_emb = self.time_emb(t.view(-1,1)) # [B, 16]
        t_emb = t_emb.view(B,1,1,-1).expand(B,L,88,-1) # [B, L,88, 16]

        pos_emb = self.pos_emb[:L] # [L, 43]
        pos_emb = pos_emb.view(1,L,1,-1).expand(B,L,88,-1) # [B, L,88, 43]

        temp = x # [B,L,P]

        x = x.view(B,L,88,1) # Prepare to cat with t_emb and pos_emb
        x = torch.cat([x,t_emb,pos_emb],dim = -1) # [B, L, 88, 60]
        x_major = music.chromatic2major(x,dim = 2) # [B, L, 52, 60]

        pitch_aware_data = self.in_p_block(x_major) # [B, L, 52, D]

        x = torch.cat([temp,pitch_aware_data.view(B,L,-1),t_emb[:,:,0,:],pos_emb[:,:,0,:]],dim=-1) # [B, L, D]

        x = self.interlayer_down(x)
        x = self.down1(x) # [B, L, D]
        x = self.up1(x) # [B, L, D]
        x = self.interlayer_up(x)
        x = x.view(B,L,52,-1) # [B, L, 52, D]

        x = torch.cat([pitch_aware_data,x],-1)

        x = self.out_p_block(x).squeeze(-1) # [B, L, 52]
        x = music.major2chromatic(x,dim = 2)

        out = self.out_p_block_88(x).squeeze(-1)


        '''
        mu = self.out_block(x)
        
        if self.learn_sigma:
            sigma = self.out_sigma_block(x)
            out = torch.cat([mu,sigma],dim=1) # that's the time dim, but the diffusion model code requires to cat on dim 1
        else:
            out = mu
            '''
        return out

class TransformerUnet(nn.Module):
    def __init__(self,d,n_blocks = 2, n_heads = 8, learn_sigma = False) -> None:
        super().__init__()
        self.learn_sigma=learn_sigma
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
            nn.Linear(88+self.d_time_emb+self.d_pos_emb,d),
            SkipConnection(
                Nonlinearity(),
                nn.Linear(d,d),
            )
        )

        '''
        self.down1 = nn.Sequential(
            *[TransformerBlock(d, n_heads) for i in range(n_blocks)]
        )
        self.up1 = nn.Sequential(
            *[TransformerBlock(d, n_heads) for i in range(n_blocks)]
        )
        '''

        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=8,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=4)

        self.out_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(d,88)
        )

        if learn_sigma:
            self.out_sigma_block = nn.Sequential(# [B,L,P,D]
                nn.Linear(d,88)
            )

        print(self)
        print(f'param count: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1_000_000}M')
            
    
    def forward(self,x,t):
        # x : [b, l=n_bar*32, p=88]
        # t : [b]
        B = x.shape[0]
        L = x.shape[1]

        t_emb = self.time_emb(t.view(-1,1)) # [B, 16]
        t_emb = t_emb.view(B,1,-1).expand(B,L,-1) # [B, L, 16]

        pos_emb = self.pos_emb[:L] # [L, 43]
        pos_emb = pos_emb.view(1,L,-1).expand(B,L,-1) # [B, L, 43]

        x = torch.cat([x,t_emb,pos_emb],dim = -1) # [B, L, D]

        x = self.in_block(x)

        '''
        x = self.down1(x) # [B, L, D]
        x = self.up1(x) # [B, L, D]
        '''
        x = self.transformer(x)

        mu = self.out_block(x)
        #out = mu
        
        if self.learn_sigma:
            sigma = self.out_sigma_block(x)
            out = torch.cat([mu,sigma],dim=1) # that's the time dim, but the diffusion model code requires to cat on dim 1
        else:
            out = mu
        
        return out