# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals
import sys
import os
import logging
import argparse
import torch
from utils import get_tasks

# Set PATHs
if "cs.nyu.edu" in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'
else:
    PATH_PREFIX = '/beegfs/aw3272/'

PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_GLOVE = PATH_PREFIX + 'raw_data/GloVe/glove.840B.300d.txt'
INFERSENT_PATH = 'infersent.allnli.pickle'

assert os.path.isfile(INFERSENT_PATH) and os.path.isfile(PATH_TO_GLOVE), 'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    # Logistics
    parser.add_argument("--cuda", help="CUDA id to use", type=int, default=0)
    parser.add_argument("--use_pytorch", help="1 to use PyTorch", type=int, default=1)
    parser.add_argument("--log_file", help="File to log to", type=str,
                        default=PATH_PREFIX+'ckpts/SentEval/infersent/log.log')
    parser.add_argument("--load_data", help="0 to read data from scratch", type=int, default=1)

    # Task options
    parser.add_argument("--tasks", help="Tasks to evaluate on, as a comma separated list", type=str)
    parser.add_argument("--max_seq_len", help="Max sequence length", type=int, default=100)

    # Model options
    parser.add_argument("--batch_size", help="Batch size to use", type=int, default=16)

    # Classifier options
    parser.add_argument("--cls_batch_size", help="Batch size to use", type=int, default=16)

    args = parser.parse_args(arguments)
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    if args.log_file:
        fileHandler = logging.FileHandler(args.log_file)
        logging.getLogger().addHandler(fileHandler)
    logging.info(args)

    # define senteval params
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': args.use_pytorch, 'kfold': 10,
            'max_seq_len': args.max_seq_len, 'batch_size': args.batch_size, 'load_data': args.load_data}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': args.cls_batch_size,
                                     'tenacity': 5, 'epoch_size': 4}

    # Load InferSent model
    params_senteval['infersent'] = torch.load(INFERSENT_PATH)
    params_senteval['infersent'].set_glove_path(PATH_TO_GLOVE)

    # Do SentEval stuff
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    tasks = get_tasks(args.tasks)
    results = se.eval(tasks)
    print(results)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
