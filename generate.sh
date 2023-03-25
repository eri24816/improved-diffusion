#!/bin/bash
conda activate music
python scripts/piano_exp.py --config config/a_vdiff.yaml 
python scripts/piano_exp.py --config config/b_vdiff.yaml 
python scripts/piano_exp.py --config config/c_vdiff.yaml 