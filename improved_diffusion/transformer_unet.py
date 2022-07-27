from turtle import forward
import torch
import torch.nn as nn
import numpy as np

Nonlinearity = nn.ReLU

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


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
        self.attention = SelfAttention(d, n_heads)
        self.linear_norm = nn.LayerNorm(d)
        self.positionwise_linear = nn.Sequential(
            nn.Linear(d, d),
            Nonlinearity(),
            nn.Linear(d, d),
        )

    def forward(self,x):
        # [B, L, D]
        x = x + self.attention(self.atten_norm(x))
        x = x + self.positionwise_linear(self.linear_norm(x))
        return x

class TransformerUnet(nn.Module):
    def __init__(self,d,n_blocks = 2, n_heads = 8, learn_sigma = False) -> None:
        super().__init__()

        # time embbeding
        self.d_time_emb = 8
        self.time_emb = nn.Linear(1, self.d_time_emb)

        # positional embedding
        self.d_pos_emb = 11
        pos = np.arange(0,4096)
        pos_emb = [(pos//(2**i))%2 for i in range(self.d_pos_emb)]
        pos_emb = np.stack(pos_emb)
        pos_emb = torch.tensor(pos_emb).transpose(0,1)
        self.register_buffer('pos_emb',pos_emb)

        # build model blocks
        self.in_block = nn.Linear(88+self.d_time_emb+self.d_pos_emb,d)
        self.down1 = nn.Sequential(
            *[TransformerBlock(d, n_heads) for i in range(n_blocks)]
        )
        self.up1 = nn.Sequential(
            *[TransformerBlock(d, n_heads) for i in range(n_blocks)]
        )
        self.out_block = nn.Sequential(
            Nonlinearity(),
            nn.Linear(d,88)
        )
        self.learn_sigma = learn_sigma
        if learn_sigma:
            self.out_sigma_block = nn.Sequential(
                Nonlinearity(),
                nn.Linear(d,88)
            )
            
    
    def forward(self,x,t):
        # x : [b, l=n_bar*32, p=88]
        # t : [b]
        batch_size = x.shape[0]
        l = x.shape[1]

        t_emb = self.time_emb(t.view(-1,1)) # [b, 8]
        t_emb = t_emb.unsqueeze(1).expand(-1,l,-1) # [b, l, 8]

        p_emb = self.pos_emb[:l] # [l, 11]
        p_emb = p_emb.unsqueeze(0).expand(batch_size,-1,-1)

        x = self.in_block(torch.cat([x,t_emb,p_emb],dim = 2))
        x = self.down1(x)
        x = self.up1(x)

        mu = self.out_block(x)
        if self.learn_sigma:
            sigma = self.out_sigma_block(x)
            out = torch.cat([mu,sigma],dim=1)
        else:
            out = mu
        return out

