import sys

name = sys.argv[1]

"""
python run_python_scapeplot.py\
 -a ../improved-diffusion/analysis/samples/{name}_sampled_mp3\
 -s ../improved-diffusion/analysis/results/{name}/ssm\
  -p ../improved-diffusion/analysis/results/{name}/scplot/\
   -j 4

python run_all_metrics.py \
-p ../improved-diffusion/analysis/results/{name}/scplot \
-o ../improved-diffusion/analysis/results/{name}/si.csv
"""
import subprocess

if __name__ == '__main__':
    subprocess.call(['python', '../MusDr/run_python_scapeplot.py',
                     '-a', f'analysis/samples/{name}_sampled_mp3',
                     '-s', f'analysis/results/{name}/ssm',
                     '-p', f'analysis/results/{name}/scplot/',
                     '-j', '4'])
    subprocess.call(['python', '../MusDr/run_all_metrics.py',
                     '-p', f'analysis/results/{name}/scplot',
                     '-o', f'analysis/results/{name}/si.csv'])