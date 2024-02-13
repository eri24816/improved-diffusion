from collections import defaultdict
import json
import sys
from chorder import Dechorder, Chord
import miditoolkit
import os
import matplotlib.pyplot as plt

import numpy as np

def release_pedal_n_times_a_bar(midi, n):
    for note in midi.instruments[0].notes:
        seg_len = 1920//n
        seg_idx = note.start//seg_len
        if note.end>seg_len*(seg_idx+1):
            note.end = int(seg_len*(seg_idx+1))

def load_all_midi(path):
    files = os.listdir(path)
    midis = []
    for file in files:
        midi = miditoolkit.midi.parser.MidiFile(os.path.join(path,file))
        midis.append(midi)
    return midis

def get_chord_prog(midi):

    if len(midi.instruments) == 0:
        return []
    release_pedal_n_times_a_bar(midi, 2)
    prog = Dechorder.dechord(midi)
    return prog

def simple_chord_name(chord:Chord):
    name = chord.root()
    if name == None:
        return None
    minor_qualities = ['m','min','dim','m7','Mm7','mm7','min7']
    if chord.quality in minor_qualities:
        name += 'm'
    return name

def get_ngram_count(chord_progs,n=3):
    '''
    return count of each n-gram class appearing in chord_progs
    '''
    chord_distribution = defaultdict(int)
    for chord_prog in chord_progs:
        chord_prog_str = [simple_chord_name(c) for c in chord_prog]
        # remove successive duplicates
        for i in range(len(chord_prog_str)-1,0,-1):
            if chord_prog_str[i] == chord_prog_str[i-1]:
                del chord_prog_str[i]
        # extract n-grams
        for i in range(len(chord_prog_str)-n+1):
            chords_str = chord_prog_str[i:i+n]
            if None in chords_str:
                continue
            chords_str = (' ').join(chords_str)
            chord_distribution[chords_str] += 1
    return chord_distribution

def calc_kld(p:dict,q:dict,smooth = 1e-3):
    '''
    Kullback-Leibler Divergence
    p and q are unnormalized counts
    '''
    for key in p:
        p[key] += smooth
    for key in q:
        q[key] += smooth
    p_sum = sum(p.values())
    q_sum = sum(q.values())
    for key in p:
        p[key] /= p_sum
    for key in q:
        q[key] /= q_sum
    kld = 0
    for key in p:
        if key not in q:
            print(key,p[key])
            continue
        kld += p[key]*np.log(p[key]/q[key])
    return kld

def calc_cosine_similarity(p:dict,q:dict):
    '''
    Cosine Similarity
    p and q are unnormalized counts
    '''
    p = p.copy()
    q = q.copy()
    p_norm = np.sqrt(sum([x**2 for x in p.values()]))
    q_norm = np.sqrt(sum([x**2 for x in q.values()]))
    for key in p:
        p[key] /= p_norm
    for key in q:
        q[key] /= q_norm
    cosine_similarity = 0
    for key in p:
        if key not in q:
            continue
        cosine_similarity += p[key]*q[key]
    return cosine_similarity

        
def bar_multiple(sequences,labels,xticks=None):
    x = np.arange(len(sequences[0]))
    width = 1/(len(sequences)+1)
    for i,histogram in enumerate(sequences):
        plt.bar(x+i*width,histogram,width,label=labels[i])
    if xticks is not None:
        plt.xticks(x,xticks)
    plt.legend()

def run_on_n_gram(dataset_chord_progs,sample_chord_progs,n,name):
    dataset_chord_distribution = get_ngram_count(dataset_chord_progs,n)
    sample_chord_distribution = get_ngram_count(sample_chord_progs,n)

    sim = calc_cosine_similarity(dataset_chord_distribution,sample_chord_distribution)

    # plot
    dataset_chord_dist_sorted = sorted(dataset_chord_distribution.items(),key=lambda x:x[1],reverse=True)
    sample_chord_dist_sorted = sorted(sample_chord_distribution.items(),key=lambda x:dataset_chord_distribution[x[0]],reverse=True)

    bar_multiple(
        [
            [x[1] for x in dataset_chord_dist_sorted[:30]],
            [x[1] for x in sample_chord_dist_sorted[:30]]
        ],
        labels=['dataset',name],xticks=[x[0].replace(' ','\n') for x in dataset_chord_dist_sorted[:30]])
    plt.title('Chord Distribution')
    plt.savefig(os.path.join('analysis/results',name,f'chord_{n}gram.png'))
    plt.close()
    return sim

def main(dataset_path, sample_path, name):
    dataset = load_all_midi(dataset_path)
    samples = load_all_midi(sample_path)
    dataset_chord_progs = [get_chord_prog(midi) for midi in dataset]
    sample_chord_progs = [get_chord_prog(midi) for midi in samples]

    sim1 = run_on_n_gram(dataset_chord_progs,sample_chord_progs,1,name)
    sim2 = run_on_n_gram(dataset_chord_progs,sample_chord_progs,2,name)
    sim3 = run_on_n_gram(dataset_chord_progs,sample_chord_progs,3,name)
    
    
    # write summary
    summary = {
        'sim1':sim1,
        'sim2':sim2,
        'sim3':sim3,
        #'dist':sample_chord_dist_sorted[:30]
        }
    print(summary)
    if not os.path.exists(os.path.join('analysis/results',name)):
        os.mkdir(os.path.join('analysis/results',name))
    json.dump(summary,open(os.path.join('analysis/results',name,'chord.json'),'w'))

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])

# usage: python analysis/metrics/chord.py analysis/samples/test_sampled analysis/samples/a42_sampled a42