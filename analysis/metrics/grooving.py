from utils.pianoroll import PianoRoll
import numpy as np

def grooving_vec(pianoroll:PianoRoll,length:int=32):
    vec = np.zeros(length)
    for onset,pitch,velocity,offset in pianoroll.iter_over_notes():
        assert onset>=0 and onset<length
        vec[onset]=1
    return vec

def hamming_distance(vec1,vec2):
    assert len(vec1)==len(vec2)
    return np.sum(vec1!=vec2)

def metric_hamming(vec1,vec2):
    return 1-hamming_distance(vec1,vec2)/len(vec1)

def metric_iou(vec1,vec2):
    return np.sum(np.logical_and(vec1,vec2))/(np.sum(np.logical_or(vec1,vec2))+1e-8)

def grooving_sim(pianoroll1:PianoRoll,pianoroll2:PianoRoll,length:int=32,metric=metric_hamming):
    vec1 = grooving_vec(pianoroll1,length)
    vec2 = grooving_vec(pianoroll2,length)
    return metric(vec1,vec2)

def grooving_sim_matrix(pianorolls,length:int=32,metric=metric_hamming):
    n = len(pianorolls)
    matrix = np.identity(n)
    for i in range(n):
        for j in range(i+1,n):
            matrix[i,j] = grooving_sim(pianorolls[i],pianorolls[j],length,metric)
            matrix[j,i] = matrix[i,j]
    return matrix

def grooving_sim_matrix_across_bars(pianoroll:PianoRoll,n_bars=16,metric=metric_hamming):
    bars = []
    for i in range(n_bars):
        bars.append(pianoroll.slice(i*32,(i+1)*32))
    return grooving_sim_matrix(bars,32,metric)

import os,sys,json
from .util import load_pianorolls_from_folder

def main(in_folder,name,samples_per_song=1):
    pianorolls = load_pianorolls_from_folder(in_folder,samples_per_song=samples_per_song)
    summary = {}
    scores = []
    for pianoroll in pianorolls:
        score = grooving_sim_matrix_across_bars(pianoroll,metric=metric_iou).mean()
        scores.append(score)
    summary['grooving'] = float(np.mean(scores))
    print(summary)
    if not os.path.exists(os.path.join('analysis/results',name)):
        os.mkdir(os.path.join('analysis/results',name))
    json.dump(summary,open(os.path.join('analysis/results',name,'grooving.json'),'w'))

    
if __name__ == '__main__':

    in_folder = sys.argv[1]
    name = sys.argv[2]
    main(in_folder,name)