from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from utils import music

from .nn_utils import Nonlinearity, SkipConnection, initialize_weight

class Sequential(nn.Sequential):
    """ Class to combine multiple models. Sequential allowing multiple inputs."""

    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def forward(self, x, *args, **kwargs):
        for i, module in enumerate(self):
            if i == 0:
                x = module(x, *args, **kwargs)
            else:
                x = module(*x, **kwargs)
            if not isinstance(x, tuple) and i != len(self) - 1:
                x = (x,)
        return x

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

class FFTransformer(nn.Module):
    def __init__(self,d,n_blocks = 4, n_heads = 8, d_cond = 0, learn_sigma = False) -> None:
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

        if condition != None:
            condition = condition.unsqueeze(1).expand(B,L,-1)
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

class TransformerWithSE(nn.Module):
    def __init__(self,wrapped, d, learn_se_input = False, output_se = False):
        super().__init__()
        self.wrapped = wrapped
        if learn_se_input:
            self.se_input = nn.Parameter(torch.zeros([d],dtype=torch.float))
        self.output_se = output_se

    def forward(self, x, se_input = None):
        if se_input == None:
            target_shape = x.shape[:-2]+(1,-1) # [...,t=1,d]
            se_input = self.se_input.expand(target_shape)
        x = torch.cat([x,se_input],-2) # cat time dim
        y = self.wrapped.forward(x)
        if self.output_se:
            return y[...,:-1,:], y[...,-1:,:]
        else:
            return y[...,:-1,:]


#https://www.cnblogs.com/wevolf/p/15188846.html
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy

    def get_position_angle_vec(position):
        # this part calculate the position In brackets
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # [:, 0::2] are all even subscripts, is dim_2i
    sinusoid_table[:, 0::2] = np.sin(np.pi*sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(np.pi*sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class TransformerUnet(nn.Module):
    def __init__(self,d1,d2,d3, depth1 = 2,depth2=2, n_heads = 8, learn_sigma = False) -> None:
        super().__init__()
        self.learn_sigma=learn_sigma
        # time embbeding
        self.d_time_emb = 16
        self.time_emb = nn.Sequential(nn.Linear(1, self.d_time_emb),Nonlinearity())

        # positional embedding
        self.d_local_pos_emb = 11#+32
        pos = np.arange(0,256)
        local_pos_emb = [(pos//(2**i))%2 for i in range(11)]#+[pos%32 == i for i in range(32)]
        local_pos_emb = np.stack(local_pos_emb)
        local_pos_emb = torch.tensor(local_pos_emb).transpose(0,1)
        self.register_buffer('local_pos_emb',local_pos_emb)

        self.d_global_pos_emb = 32
        global_pos_emb = get_sinusoid_encoding_table(400,self.d_global_pos_emb)
        self.register_buffer('global_pos_emb',global_pos_emb)

        # build model blocks
        self.in_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(88+self.d_time_emb+self.d_local_pos_emb,d1),
            SkipConnection(
                Nonlinearity(),
                nn.Linear(d1,d1),
            )
        )

        encoder_layer1 = nn.TransformerEncoderLayer(d_model=d1, nhead=n_heads,batch_first=True)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=d2, nhead=n_heads,batch_first=True)

        self.down1 = TransformerWithSE(nn.TransformerEncoder(encoder_layer1,num_layers=depth1),d1,True,True)

        self.down2 = nn.Sequential( 
            nn.Linear(d1+self.d_global_pos_emb,d2),
            TransformerWithSE(nn.TransformerEncoder(encoder_layer2,num_layers=depth2),d2,True,True)
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(d2,d3),
            Nonlinearity(),
            nn.Linear(d3,d2),
            Nonlinearity(),
        )

        self.up2 = Sequential( 
            TransformerWithSE(nn.TransformerEncoder(encoder_layer2,num_layers=depth2),d2,False,False),
            nn.Linear(d2,d1),
        )

        self.up1 = TransformerWithSE(nn.TransformerEncoder(encoder_layer1,num_layers=depth1),d1,False,False)

        self.out_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(d1,88)
        )

        if learn_sigma:
            self.out_sigma_block = nn.Sequential(# [B,L,P,D]
                nn.Linear(d1,88)
            )

        #print(self)
        #print(f'param count: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1_000_000}M')
            
    
    def forward(self,x,t):
        # x : [b, l=n_bar*32, p=88]
        # t : [b]
        B = x.shape[0]
        L = x.shape[1]
        N_BAR = L//32

        t_emb = self.time_emb(t.view(-1,1)) # [B, 16]
        t_emb = t_emb.view(B,1,-1).expand(B,L,-1) # [B, L, 16]

        pos_emb = self.local_pos_emb[:32] # [32, 43]
        pos_emb = pos_emb.view(1,1,32,-1).expand(B,N_BAR,32,-1).reshape(B,L,-1) # [B, L, 43]

        x = torch.cat([x,t_emb,pos_emb],dim = -1) # [B, L, D]

        '''
        Down path
        '''

        x = self.in_block(x) # [B, L, d1]

        # Split bars
        x = x.view(B*N_BAR,32,-1) # [B*n_bar, 32, d1]
        skip1, x = self.down1(x) # skip1: [B*n_bar, 32, d1], x: [B*n_bar, 1, d1]

        # Bar embedding
        x = x.view(B,N_BAR,-1) # [B, N_BAR, d1]
        global_pos_emb = self.global_pos_emb[:,:N_BAR] # [N_BAR, d1 + d_global_pos_emb]
        global_pos_emb = global_pos_emb.expand(B,N_BAR,-1) # [B, N_BAR, d1 + d_global_pos_emb]
        x = torch.cat([x,global_pos_emb],dim = -1) # [B, N_BAR, d1 + d_global_pos_emb]
        skip2, x = self.down2(x) # skip2: [B, n_bar, d2], x: [B, 1, d2]

        x = self.bottleneck(x) # [B, 1, d2]

        '''
        Up path
        '''

        x= self.up2(skip2,x) # [B, n_bar, d1]
        x = x.view(B*N_BAR,1,-1) # [B*n_bar, 1, d1]

        x= self.up1(skip1,se_input = x) # [B*n_bar, 32, d1]

        x = x.reshape(B,L,-1) # [B, L, d1]

        mu = self.out_block(x)
        
        if self.learn_sigma:
            sigma = self.out_sigma_block(x)
            out = torch.cat([mu,sigma],dim=1) # that's the time dim, but the diffusion model code requires to cat on dim 1
        else:
            out = mu
        
        return out