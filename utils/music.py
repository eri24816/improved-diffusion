from dataclasses import dataclass
from functools import reduce
from typing import List
import numpy as np
import torch
import re

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

class Chord:
    def __init__(self, chord_name) -> None:
        self.base_chord = chord_name[0] # C, D, E, F, G, A, B
        self.base_chord_ord = {'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}[self.base_chord]
        self.is_minor = False
        self.is_major7 = False
        self.is_minor7 = False
        self.is_sus2 = False
        self.is_sus4 = False
        self.is_add2 = False
        self.is_add4 = False
        i = 1
        while i < len(chord_name):
            if chord_name[i] == '#':
                self.base_chord+='#'
                self.base_chord_ord += 1
            elif chord_name[i] == 'b':
                self.base_chord+='b'
                self.base_chord_ord -= 1
            elif chord_name[i] == 'm':
                self.is_minor = True
            elif chord_name[i] == 'M':
                assert chord_name[i+1] == '7'
                self.is_major7 = True
                i += 1
            elif chord_name[i] == '7':
                self.is_minor7 = True
            elif chord_name[i:i+4] == 'sus2':
                self.is_sus2 = True
                i += 3
            elif chord_name[i:i+4] == 'sus4':
                self.is_sus4 = True
                i += 3
            elif chord_name[i:i+4] == 'add2':
                self.is_add2 = True
                i += 3
            elif chord_name[i:i+4] == 'add4':
                self.is_add4 = True
                i += 3
            else:
                raise ValueError('Unknown chord name '+chord_name)
            i += 1

    def to_chroma(self,semi_shift=0):
        '''
        args: chord_name: str, semi_shift: int
        return: chroma: torch.Tensor [12]. chroma[0] is C
        '''
        if not self.is_minor:
            chroma = torch.tensor([1,0,0,0,1,0,0,1,0,0,0,0])
        else:
            chroma = torch.tensor([1,0,0,1,0,0,0,1,0,0,0,0])
        if self.is_sus2:
            chroma[2] = 1
            chroma[3] = 0
            chroma[4] = 0
        if self.is_sus4:
            chroma[5] = 1
            chroma[3] = 0
            chroma[4] = 0
        if self.is_add2:
            chroma[2] = 1
        if self.is_add4:
            chroma[5] = 1
        if self.is_major7:
            chroma[11] = 1
        if self.is_minor7:
            chroma[10] = 1
        chroma = torch.roll(chroma,shifts=self.base_chord_ord + semi_shift,dims=-1)
        return chroma.float()

def expand_chord_sequence(chord_sequence:str, num_repeat_interleave=1) -> str:
    '''
    example chord: 'Am F C (G E)' repeat_interleave=2
    return: 'Am Am F F C C G E'
    '''
    chord_list = []
    for token in re.finditer(r'([A-Za-z0-9]+)|\((.*?)\)', chord_sequence):
        if token.group(1):
            chord = [token.group(1)]
            repeats = num_repeat_interleave
        else:
            print(token.group(2))
            chord = token.group(2).split(' ')
            assert num_repeat_interleave % len(chord) == 0
            repeats = num_repeat_interleave//len(chord)
        for c in chord:
            chord_list += [c]*repeats

    return ' '.join(chord_list)

def chord_sequence_to_chords(chord_sequence:str, num_repeat_interleave=1) -> List[Chord]:
    '''
    example chord: 'Am F C (G E)'
    return: [Chord('Am'), Chord('Am'), Chord('F'), Chord('F'), Chord('C'), Chord('C'), Chord('G'), Chord('E')]
    '''
    chord_sequence = expand_chord_sequence(chord_sequence, num_repeat_interleave)
    chords = []
    for chord_name in chord_sequence.split(' '):
        chords.append(Chord(chord_name))
    return chords

def generate_chroma_map(chords:List[Chord], num_segments:int) -> torch.Tensor:
    chroma_map = []
    for i in range(num_segments):
        chroma_map.append(chords[i%len(chords)].to_chroma())
    chroma_map = torch.stack(chroma_map,dim=0)
    return chroma_map
