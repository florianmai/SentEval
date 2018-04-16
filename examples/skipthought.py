# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

"""
Example of file for SkipThought in SentEval
"""
import os
import pdb
import sys
import argparse
import logging

from utils import get_tasks, write_results

# Set PATHs
if "cs.nyu.edu" in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'
else:
    PATH_PREFIX = '/beegfs/aw3272/'
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_SKIPTHOUGHT = PATH_PREFIX + 'models/skip-thoughts/'

assert PATH_TO_SKIPTHOUGHT != '', 'Download skipthought and set correct PATH'

# import skipthought and Senteval
sys.path.insert(0, PATH_TO_SKIPTHOUGHT)
import skipthoughts
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def prepare(params, samples):
    return

def batcher(params, batch):
    #batch = [str(' '.join(sent), errors="ignore") if sent != [] else '.' for sent in batch]
    try:
        #encoded_batch = [' '.join(sent).encode('utf-8') if sent != [] else '.' for sent in batch]
        encoded_batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
        embeddings = skipthoughts.encode(params['encoder'], encoded_batch, verbose=False, use_eos=True)
    except:
        pdb.set_trace()
    return embeddings


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    # Logistics
    parser.add_argument("--seed", help="Random seed", type=int, default=19)
    parser.add_argument("--cuda", help="CUDA id to use", type=int, default=0)
    parser.add_argument("--use_pytorch", help="1 to use PyTorch", type=int, default=1)
    parser.add_argument("--out_dir", help="Dir to write preds to", type=str, default='')
    parser.add_argument("--log_file", help="File to log to", type=str)
    parser.add_argument("--load_data", help="0 to read data from scratch", type=int, default=1)

    # Task options
    parser.add_argument("--tasks", help="Tasks to evaluate on, as a comma separated list", type=str)
    parser.add_argument("--max_seq_len", help="Max sequence length", type=int, default=40)

    # Model options
    parser.add_argument("--batch_size", help="Batch size to use", type=int, default=64)
    parser.add_argument("--dict_file", help="File to load dict from", type=str)
    parser.add_argument("--model_file", help="File to load model from", type=str)

    # Classifier options
    parser.add_argument("--cls_batch_size", help="Batch size to use for classifiers",
                        type=int, default=64)

    args = parser.parse_args(arguments)
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    if args.log_file:
        fileHandler = logging.FileHandler(args.log_file)
        logging.getLogger().addHandler(fileHandler)
    logging.info(args)

    # Set params for SentEval
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': args.use_pytorch, 'kfold': 10,
            'max_seq_len': args.max_seq_len, 'batch_size': args.batch_size, 'load_data': args.load_data,
            'seed': args.seed}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': args.cls_batch_size,
            'tenacity': 5, 'epoch_size': 4, 'cudaEfficient': True}
    params_senteval['encoder'] = skipthoughts.load_model()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    tasks = get_tasks(args.tasks)
    results = se.eval(tasks)
    if args.out_dir:
        write_results(results, args.out_dir)
    if not args.log_file:
        print(results)
    else:
        logging.info(results)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
