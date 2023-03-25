import os,sys
from utils.pianoroll import PianoRoll
# get args

in_path = sys.argv[1]
out_path = sys.argv[2]

PianoRoll.from_midi(in_path).to_img(out_path)