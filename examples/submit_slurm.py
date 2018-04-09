''' Script for submitting scripts to slurm '''
import os
import pdb
import time
import subprocess
import datetime

if 'cs.nyu.edu' in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang'
else:
    PATH_PREFIX = '/beegfs/aw3272'

proj_name = 'SentEval'
exp_name = 'gensen' # need to make the folders if don't exist
run_name = 'benchmark_v2'
error_file = '%s/ckpts/%s/%s/%s.err' % (PATH_PREFIX, proj_name, exp_name, run_name)
out_file = '%s/ckpts/%s/%s/%s.out' % (PATH_PREFIX, proj_name, exp_name, run_name)
log_file = '%s/ckpts/%s/%s/%s.log' % (PATH_PREFIX, proj_name, exp_name, run_name)
slurm_args = ['-J', exp_name, '-e', error_file, '-o', out_file, '-t', '2-00:00',
        '--gres=gpu:1080ti:1', '--mail-type=end', '--mail-user=aw3272@nyu.edu']

model = 'gensen'
tasks = 'MNLI'
use_pytorch = '1'
cls_batch_size = '32'

py_args = [model, tasks, log_file, use_pytorch, cls_batch_size]

slurm = 0

if slurm:
    cmd = ['sbatch'] + slurm_args + ['run_stuff.sh'] + py_args
else:
    cmd = ['./run_stuff.sh'] + py_args
print(' '.join(cmd))
subprocess.call(cmd)
time.sleep(10)

############################
# MAKE SURE YOU CHANGE:
#   - exp_name
#   - run_name
#   - other parameters as needed
############################
