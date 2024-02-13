import math
import os
import random
from utils.pianoroll import PianoRoll
import sys

def load_pianorolls_from_folder(path,samples_per_song=1,num_samples=None):

    songs = []
    
    files = os.listdir(path)
    # sort by name
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    files.sort(key=lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] )
    for file in files:
        if file.endswith(".mid") or file.endswith(".midi"):
            songs.append(PianoRoll.from_midi(os.path.join(path, file)))
        if file.endswith(".json"):
            try:
                songs.append(PianoRoll.load(os.path.join(path, file)))
            except:
                print('warning: failed to load',os.path.join(path, file))

    if num_samples is not None:
        samples_per_song = math.ceil(num_samples/len(songs))

    samples = []
    for song in songs:
        for i in range(samples_per_song):
            start = random.randint(0,max(0,song.duration//32-16))*32
            samples.append(song.slice(start,start+32*16))

    print(len(samples),num_samples)
    if len(samples) != num_samples:
        random.shuffle(samples)
        if num_samples is not None:
            samples = samples[:num_samples]
    
    return samples

if __name__ == '__main__':
    path = sys.argv[1]
    num_samples = int(sys.argv[2])
    samples = load_pianorolls_from_folder(path,num_samples=num_samples)

    if path.endswith('/'):
        path = path[:-1]
    path_head, path_tail = os.path.split(path)
    print('saving to',os.path.join(path_head,f'{path_tail}_sampled'))
    os.mkdir(os.path.join(path_head,f'{path_tail}_sampled'))
    for i,sample in enumerate(samples):
        sample:PianoRoll
        sample.to_midi(os.path.join(path_head,f'{path_tail}_sampled',f'{i}_{str(sample)}.mid'),bpm=105)


# usage: python scripts/sample_from_folder.py /screamlab/home/eri24816/improved-diffusion/log/experiments/a42  256
# python scripts/sample_from_folder.py /screamlab/home/eri24816/compound-word-transformer/workspace/uncond/cp-linear/gen_midis/30  256