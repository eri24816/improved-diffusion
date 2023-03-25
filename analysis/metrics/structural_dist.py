import random
from typing import List
import os, sys
from utils.pianoroll import PianoRoll
import numpy as np
import matplotlib.pyplot as plt
import json
from .util import load_pianorolls_from_folder

high_clamp = 51
low_clamp = 39
pitch_weight = lambda pitch: min(1,max(1e-8,1-(pitch-low_clamp)/(high_clamp-low_clamp)))
#pitch_weight = lambda pitch: 1
#pitch_weight = lambda pitch: 1 if low_clamp<39 else 1e-8
local_shift_search_order = [0,1,-1,2,-2,12,-12,3,-3,4,-4,24,-24,5,-5,6,-6,36,-36]
local_shift_losses = [0,1,1,2,2,2,2,3,3,4,4,4,4,5,5,6,6,6,6]
global_shift_losses = [0,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
def harmony_dist_asym(a:List,b:List):
    b_dense = [0]*88
    for note in b:
        b_dense[note] = 1

    min_loss = 1000000
    for global_shift in range(-12,13):
        loss = 0
        loss += global_shift_losses[abs(global_shift)]
        for note in a:
            note_gs = note + global_shift
            for local_shift, local_shift_loss in zip(local_shift_search_order,local_shift_losses):
                note_gs_ls = note_gs + local_shift
                if note_gs_ls>=0 and note_gs_ls<88 and b_dense[note_gs_ls]:
                    loss += local_shift_loss*pitch_weight(note)
                    break
            else:
                loss += 6*pitch_weight(note)
        min_loss = min(min_loss,loss)
    return min_loss

def harmony_dist(a:List,b:List):
    #return max(harmony_dist_asym(a,b),harmony_dist_asym(b,a))
    return (harmony_dist_asym(a,b)+harmony_dist_asym(b,a))/2

def structural_dist(a:PianoRoll,b:PianoRoll,quantize_resolution=32):

    quantize_factor = 32//quantize_resolution
    #quantize
    for note in a.notes:
        note.onset = note.onset//quantize_factor*quantize_factor

    time_steps_with_onsets = set()
    for onset,pitch,velocity,offset in a.iter_over_notes():
        time_steps_with_onsets.add(onset)
    for onset,pitch,velocity,offset in b.iter_over_notes():
        time_steps_with_onsets.add(onset)
    
    harmony_pairs_ab = {onset:{'a':[],'b':[]}for onset in time_steps_with_onsets}
    harmony_pairs_ba = {onset:{'a':[],'b':[]}for onset in time_steps_with_onsets}
    denom_a = 0
    denom_b = 0

    a_already_pressed = set()
    for onset,pitch,velocity,offset in a.iter_over_notes():
        pitch -= 21

        harmony_pairs_ba[onset]['a'].append(pitch)
        if pitch not in a_already_pressed:
            harmony_pairs_ab[onset]['a'].append(pitch)
            a_already_pressed.add(pitch)
            denom_a += pitch_weight(pitch)

    b_already_pressed = set()
    for onset,pitch,velocity,offset in b.iter_over_notes():
        pitch -= 21

        harmony_pairs_ab[onset]['b'].append(pitch)
        if pitch not in b_already_pressed:
            harmony_pairs_ba[onset]['b'].append(pitch)
            b_already_pressed.add(pitch)
            denom_b += pitch_weight(pitch)

    dist = 0
    for pair in harmony_pairs_ab.values():
        dist += harmony_dist_asym(pair['a'],pair['b'])/(denom_a+1e-8)
    for pair in harmony_pairs_ba.values():
        dist += harmony_dist_asym(pair['b'],pair['a'])/(denom_b+1e-8)


    return dist

def structural_dist_matrix(a:PianoRoll,quantize_resolution=32):
    matrix = np.zeros((16,16))
    for i in range(16):
        for j in range(i+1,16):
            matrix[i,j] = structural_dist(a.slice(i*32,(i+1)*32),a.slice(j*32,(j+1)*32),quantize_resolution)
            matrix[j,i] = matrix[i,j]
    return matrix

def nonempty_mask(a:PianoRoll,min_num_notes=4):
    mask = np.ones((16,16))
    for i in range(16):
        target = a.slice(i*32,(i+1)*32)
        if len(target.notes)<min_num_notes:
            mask[i,:] = 0
            mask[:,i] = 0
    return mask


# metrics based on dist matrix
def diag_entries_hist(dist_mats_and_masks,diag = 1):
    dists = []
    denom = 0
    for dist_mat, mask in dist_mats_and_masks:
        dist_mat:np.ndarray
        dists.extend(dist_mat.diagonal(diag)[mask.diagonal(diag)==1].tolist())
        denom+=sum(mask.diagonal(diag))
    #histogram
    histogram = [0.]*20
    for i in range(20):
        histogram[i] = sum([d<i+1 and d>=i for d in dists])

    for i in range(20):
        histogram[i]/=(denom+1e-8)

    return histogram, denom

def count_repititions(dist_mats_and_masks,diag=1,min_len=1,threshold=2):
    count = 0
    denom = 0
    for dist_mat, mask in dist_mats_and_masks:
        dist_mat:np.ndarray
        mask:np.ndarray
        dists = dist_mat.diagonal(diag)
        masks = mask.diagonal(diag)
        local_count = 0
        for i in range(len(dists)):
            if masks[i]:
                denom+=1
            if masks[i] and dists[i]<threshold:
                local_count+=1
            else:
                if local_count>=min_len:
                    count+=local_count
                local_count = 0
        if local_count>=min_len:
            count+=local_count
    return count/(denom+1e-8)

def analyze_stucture(songs:List[PianoRoll],samples_per_song=1):
    samples = []
    for song in songs:
        for i in range(samples_per_song):
            start = random.randint(0,max(0,song.duration//32-16))*32
            samples.append(song.slice(start,start+32*16))

    dist_mats_and_masks = []
    for sample in samples:
        dist_mat = structural_dist_matrix(sample)
        mask = nonempty_mask(sample)
        dist_mat[mask==0]=1000000
        dist_mats_and_masks.append((dist_mat,mask))

    diag_hist = [[]for _ in range(16)]
    diag_hist_denoms = [0.]*16
    for i in range(1,16):
        diag_hist[i], diag_hist_denoms[i] = diag_entries_hist(dist_mats_and_masks,diag=i)

    rep_counts = {}
    threshold = 2
    rep_params = {
        'AA':{'diag':1,'min_len':1},
        'A1A':{'diag':2,'min_len':1},
        'A3A':{'diag':4,'min_len':1},
        'AAA':{'diag':1,'min_len':2},
        'ABAB':{'diag':2,'min_len':2},
        'AB2AB':{'diag':4,'min_len':2},
    }
    for name, params in rep_params.items():
        rep_counts[name] = count_repititions(dist_mats_and_masks,diag=params['diag'],min_len=params['min_len'],threshold=threshold)

    return diag_hist, diag_hist_denoms, rep_counts

def analyze_stucture_from_folder(path,samples_per_song=1):
    samples = load_pianorolls_from_folder(path)
    return analyze_stucture(samples,samples_per_song)

def summarize(analyzed,threshold=2):
    diag_hist, diag_hist_denoms, rep_counts = analyzed
    summary = {}
    for diag in [1,2,4]:
        summary['diag'+str(diag)] = sum([diag_hist[diag][d] for d in range(threshold)])

    diag5plus = 0
    total_weight = 0
    for diag in range(5,16):
        weight = diag_hist_denoms[diag]
        diag5plus += sum([diag_hist[diag][d] for d in range(threshold)])*weight
        total_weight += weight
    summary['diag5+'] = diag5plus/total_weight
    
    for name, count in rep_counts.items():
        summary[name] = count
    return summary


def main(in_folder,name,samples_per_song=1):
    analyzed = analyze_stucture_from_folder(in_folder,samples_per_song=samples_per_song)
    summary = summarize(analyzed)
    print(summary)
    if not os.path.exists(os.path.join('analysis/results',name)):
        os.mkdir(os.path.join('analysis/results',name))
    json.dump(summary,open(os.path.join('analysis/results',name,'structure.json'),'w'))

if __name__ == '__main__':
    in_folder = sys.argv[1]
    name = sys.argv[2]
    main(in_folder,name)

# usage: python metrics/structural_dist.py aa/output/ cp32