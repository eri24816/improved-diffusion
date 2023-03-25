import os,sys
from utils.pianoroll import PianoRoll
# get args

in_path = sys.argv[1]
out_path = sys.argv[2]

PianoRoll.from_midi(in_path).save_to_pretty_score(out_path,position_weight=1,mode='separate',make_pretty_voice=False)
