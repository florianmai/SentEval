''' Script for submitting scripts to slurm '''
import os
import pdb
import time
import subprocess
import datetime

if 'cs.nyu.edu' in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang'
    DEVICE = '1080ti'
else:
    PATH_PREFIX = '/beegfs/aw3272'
    DEVICE = 'p40'

proj_name = 'SentEval'
model = 'skipthought'
exp_name = model # need to make the folders if don't exist
run_name = 'benchmark_v4'
run_dir = "%s/ckpts/%s/%s/%s" % (PATH_PREFIX, proj_name, exp_name, run_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
error_file = '%s/ckpts/%s/%s/%s/sbatch.err' % (PATH_PREFIX, proj_name, exp_name, run_name)
out_file = '%s/ckpts/%s/%s/%s/sbatch.out' % (PATH_PREFIX, proj_name, exp_name, run_name)
log_file = '%s/ckpts/%s/%s/%s/log.log' % (PATH_PREFIX, proj_name, exp_name, run_name)
slurm_args = ['-J', exp_name, '-e', error_file, '-o', out_file, '-t', '2-00:00',
        '--gres=gpu:%s:1' % DEVICE, '--mail-type=end', '--mail-user=aw3272@nyu.edu']

tasks = 'benchmark'
use_pytorch = '1'
batch_size = '128'
cls_batch_size = '128'
max_seq_len = '40'

py_args = [model, tasks, log_file, use_pytorch, cls_batch_size, batch_size, max_seq_len, run_dir]

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
