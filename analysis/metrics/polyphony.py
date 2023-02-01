import torch
import matplotlib.pyplot as plt
from utils.pianoroll import PianoRoll, PianoRollDataset
from tqdm import tqdm
import numpy as np

velocity_mapping = lambda x: x**0.1

def calc_polyphony(pr:torch.Tensor,velocity_mapping = lambda x: x**0.1):
    """Calculate the polyphony of a one-bar pianoroll.

    Args:
        pr (torch.Tensor): A pianoroll tensor with shape (32,88). Values are in [0,127].

    Returns:
        int: The polyphony of the pianoroll.
    """
    assert pr.shape == (32,88)
    pr = velocity_mapping(pr/128) # [32,88]
    #pr = (pr>0).float()#===============================
    end_dist_list = []
    scanner = torch.zeros(88,device=pr.device)
    for i in range(31,-1,-1):
        scanner = scanner + 1
        end_dist_list += [scanner]
        current = pr[i]
        scanner = scanner * (1-current)

    end_dist = torch.stack(list(reversed(end_dist_list)),dim=0) # [32,88]
    weights = end_dist

    weighted_pr = pr * weights # [32,88]
    polyphony = torch.mean(weighted_pr)*88
    return polyphony

if __name__ == '__main__':
    #　Load the data
    ds = PianoRollDataset('/screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll_split/train')
    #　Calculate the polyphony
    bins = np.zeros(128)
    polyphony_list = []
    for pr in tqdm(ds.pianorolls):
        pr = pr.to_tensor()
        pr = pr.view(-1,32,88)
        for bar in pr:
            polyphony = calc_polyphony(bar)
            bins[int(polyphony*128)]+=1
            polyphony_list.append(polyphony)

    plt.bar(range(128),bins)
    plt.savefig('polyphony.png')
    print(np.quantile(polyphony_list, np.arange(0,1.1,0.1)))