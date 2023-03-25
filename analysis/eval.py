import numpy as np
from utils.io_util import json_load, json_dump
import os, sys
from utils.pianoroll import PianoRoll
from metrics import grooving, structural_dist

if __name__ == '__main__':
    in_folder = sys.argv[1]
    name = sys.argv[2]
    samples_per_song = int(sys.argv[3])
    grooving.main(in_folder, name, samples_per_song)
    structural_dist.main(in_folder, name, samples_per_song)

# python analysis/eval.py log/experiments/b28 b28 1