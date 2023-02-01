import torch
import matplotlib.pyplot as plt
from utils.pianoroll import PianoRoll, PianoRollDataset
from tqdm import tqdm
import numpy as np

bins = np.zeros(128)

velocity_mapping = lambda x: (x<0.2)*4.5*x+(x>0.2)*(0.9+0.125*(x-0.2))
if __name__ == '__main__':
    #　Load the data
    ds = PianoRollDataset('/screamlab/home/eri24816/pianoroll_dataset/data/dataset_1/pianoroll_split/val')
    
    for pr in tqdm(ds.pianorolls):
        nonzero_count = 0
        for note in pr.notes:
            velocity = int(velocity_mapping(note.velocity/128)*128)
            bins[velocity]+=1
            nonzero_count+=1
        zero_count = 88*pr.duration - nonzero_count
        bins[0]+=zero_count

    plt.bar(range(128),np.log(bins+1))
    plt.savefig('velocity_distribution.png')
