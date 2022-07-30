from dataclasses import dataclass
from functools import reduce
import numpy as np
import torch

MAJOR_SCALE =  np.concatenate(
    [np.array([0,2])]+
    [np.array([0,2,4,5,7,9,11])+3+oct*12 for oct in range(7)]+
    [np.array([87])]
    )
# [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 29, 31, 32, 34, 36, 38, 39, 41, 43, 44, 46, 48, 50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87]

INV_MAJOR_SCALE = [len(MAJOR_SCALE)]*88 # len(MAJOR_SCALE) means null
for major, semi in enumerate(MAJOR_SCALE):
    INV_MAJOR_SCALE[semi] = major

UP_SEMI = [88]*len(MAJOR_SCALE) # 88 means null

for i in range(len(MAJOR_SCALE)-1):
    n=MAJOR_SCALE[i+1]-MAJOR_SCALE[i]
    if n > 1:
        UP_SEMI[i] = MAJOR_SCALE[i]+1


def chromatic2major(x,dim = -1):
    # piano pitch number
    assert x.shape[dim]==88

    shape = list(x.shape)
    shape1 = list(x.shape)
    shape11 = [1]*len(x.shape)
    shape[dim] = -1
    shape1[dim] = 1
    shape11[dim] = -1


    idx = torch.tensor(MAJOR_SCALE).view(shape11).expand(*shape).to(x.device)
    major = torch.gather(x,dim,idx)
    '''

    idx = torch.tensor(UP_SEMI).view(shape11).expand(*shape).to(x.device)
    x = torch.cat([x,torch.zeros(shape1,device=x.device)],dim)
    up_semi = torch.gather(x,dim,idx)
    '''
    res = major #+ up_semi

    return res

def major2chromatic(x,dim=-1):
    # piano pitch number
    assert x.shape[dim]==52 # white keys

    shape = list(x.shape)
    shape1 = list(x.shape)
    shape11 = [1]*len(x.shape)
    shape[dim] = -1
    shape1[dim] = 1
    shape11[dim] = -1


    idx = torch.tensor(INV_MAJOR_SCALE).view(shape11).expand(*shape).to(x.device)
    x = torch.cat([x,torch.zeros(shape1,device=x.device)],dim)
    res = torch.gather(x,dim,idx)
    return res


from utils.pianoroll import PianoRollDataset
import matplotlib.pyplot as plt
d = PianoRollDataset('/screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll',64)
print(len(d))
'''
s = d[100]
print(((major2chromatic(chromatic2major(s)) - s)**2).sum())
plt.imshow(s)
plt.savefig('a.png')
plt.imshow(chromatic2major(s))
plt.savefig('b.png')
'''