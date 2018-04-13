''' Utils to use across model runs '''
import os
import sys
import ipdb as pdb
import numpy as np

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

def write_results(results, out_dir):
    for task, task_results in results.items():
        if 'preds' not in task_results:
            continue
        preds = task_results['preds']
        with open(os.path.join(out_dir, "%s_preds.tsv" % task), 'w') as out_fh:
            out_fh.write("index\tprediction\n")
            if isinstance(preds, np.ndarray):
                preds = preds.squeeze().tolist()
            for idx, pred in enumerate(preds):
                out_fh.write("%d\t%.3f\n" % (idx, pred))

        if task == 'MNLI':
            preds = task_results['mismatched_preds']
            with open(os.path.join(out_dir, "%s_mismatched_preds.tsv" % task), 'w') as out_fh:
                out_fh.write("index\tprediction\n")
                if isinstance(preds, np.ndarray):
                    preds = preds.squeeze().tolist()
                for idx, pred in enumerate(preds):
                    out_fh.write("%d\t%.3f\n" % (idx, pred))
            preds = task_results['diagnostic_preds']
            with open(os.path.join(out_dir, "diagnostic_preds.tsv"), 'w') as out_fh:
                out_fh.write("index\tprediction\n")
                if isinstance(preds, np.ndarray):
                    preds = preds.squeeze().tolist()
                for idx, pred in enumerate(preds):
                    out_fh.write("%d\t%.3f\n" % (idx, pred))

