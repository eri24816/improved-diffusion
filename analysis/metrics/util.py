import os
from utils.pianoroll import PianoRoll
import random

def load_pianorolls_from_folder(path,samples_per_song=1):
    songs = []
    
    for file in os.listdir(path):
        if file.endswith(".mid") or file.endswith(".midi"):
            songs.append(PianoRoll.from_midi(os.path.join(path, file)))
        if file.endswith(".json"):
            try:
                songs.append(PianoRoll.load(os.path.join(path, file)))
            except:
                print('warning: failed to load',os.path.join(path, file))

    samples = []
    for song in songs:
        for i in range(samples_per_song):
            start = random.randint(0,max(0,song.duration//32-16))*32
            samples.append(song.slice(start,start+32*16))
    
    return samples
    