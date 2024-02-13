import subprocess
import numpy as np
from utils.io_util import json_load, json_dump
import os, sys
from utils.pianoroll import PianoRoll
from metrics import grooving, chord

if __name__ == '__main__':

    name = sys.argv[1]
    grooving.main(f'analysis/samples/{name}_sampled', name, 1)
    chord.main(f'analysis/samples/test_sampled',f'analysis/samples/{name}_sampled',name)

    # subprocess.call(['python', '../MusDr/run_python_scapeplot.py',
    #                  '-a', f'analysis/samples/{name}_sampled_mp3',
    #                  '-s', f'analysis/results/{name}/ssm',
    #                  '-p', f'analysis/results/{name}/scplot/',
    #                  '-j', '4'])
    # subprocess.call(['python', '../MusDr/run_all_metrics.py',
    #                  '-p', f'analysis/results/{name}/scplot',
    #                  '-o', f'analysis/results/{name}/si.csv'])