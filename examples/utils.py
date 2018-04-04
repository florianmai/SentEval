''' Utils to use across model runs '''
import sys
import pdb

PATH_SENTEVAL = '../'
sys.path.insert(0, PATH_SENTEVAL)
from senteval.engine import ALL_TASKS, BENCHMARK_TASKS

def get_tasks(task_str):
    if task_str == 'all':
        tasks = ALL_TASKS
    elif task_str == 'benchmark':
        tasks = BENCHMARK_TASKS
    else:
        tasks = task_str.split(',')
    return tasks
