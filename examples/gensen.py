#!/usr/bin/env python
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import os
import pdb
import sys
import torch
import logging
import argparse
from utils import get_tasks

if "cs.nyu.edu" in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'
else:
    PATH_PREFIX = '/beegfs/aw3272/'

# Set PATHs
PATH_GENSEN = PATH_PREFIX + '/models/GenSen-master/'
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval
sys.path.insert(0, PATH_GENSEN)
from gensen import GenSen, GenSenSingle

STRATEGY = "best"

def prepare(params, samples):
    print('Preparing task : %s ' % (params.current_task))
    vocab = set()
    for sample in samples:
        if params.current_task != 'TREC':
            sample = ' '.join(sample).lower().split()
        else:
            sample = ' '.join(sample).split()
        for word in sample:
            if word not in vocab:
                vocab.add(word)
    for tok in ['<s>', '<pad>', '<unk>', '</s>']:
        vocab.add(tok)

    # If you want to turn off vocab expansion just comment out the below line.
    params['gensen'].vocab_expansion(vocab)


def batcher(params, batch):
    # batch contains list of words
    max_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'ImageCaptionRetrieval']
    if STRATEGY == 'best':
        if params.current_task in max_tasks:
            strategy = 'max'
        else:
            strategy = 'last'
    else:
        strategy = STRATEGY

    sentences = [' '.join(s).lower() for s in batch]
    _, embeddings = params['gensen'].get_representation(sentences, pool=strategy, return_numpy=True)
    return embeddings

def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    # Logistics
    parser.add_argument("--gpu_id", help="gpu id to use", type=int, default=0)
    parser.add_argument("--use_pytorch", help="1 to use PyTorch", type=int, default=0)
    parser.add_argument("--log_file", help="File to log to", type=str,
                        default=PATH_PREFIX + 'ckpts/SentEval/gensen/log.log')
    parser.add_argument("--load_data", help="0 to read data from scratch", type=int, default=1)

    # Model options
    parser.add_argument("--batch_size", help="Batch size to use", type=int, default=16)
    parser.add_argument("--folder_path", help="path to model folder", default=PATH_GENSEN + 'data/models')
    parser.add_argument("--prefix_1", help="prefix to model 1", default='nli_large_bothskip_parse')
    parser.add_argument("--prefix_2", help="prefix to model 2", default='nli_large_bothskip')
    parser.add_argument("--pretrain", help="path to pretrained vectors",
                        default=PATH_GENSEN + 'data/embedding/glove.840B.300d.h5')
    # NOTE: To decide the pooling strategy for a new model, note down the validation set scores below.
    parser.add_argument("--strategy", help="Approach to create sentence embedding last/max/best", default="best")

    # Task options
    parser.add_argument("--tasks", help="Tasks to evaluate on, as a comma separated list", type=str)
    parser.add_argument("--max_seq_len", help="Max sequence length", type=int, default=100)


    # Classifier options
    parser.add_argument("--cls_batch_size", help="Batch size to use for the classifier", type=int,
                        default=16)

    args = parser.parse_args(arguments)
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    if args.log_file:
        fileHandler = logging.FileHandler(args.log_file)
        logging.getLogger().addHandler(fileHandler)
    logging.info(args)
    torch.cuda.set_device(args.gpu_id)

    # Set up SentEval
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': args.use_pytorch, 'kfold': 10,
            'max_seq_len': args.max_seq_len, 'batch_size': args.batch_size, 'load_data': args.load_data}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': args.cls_batch_size,
                                     'tenacity': 5, 'epoch_size': 4}

    # Load model
    gensen_1 = GenSenSingle(model_folder=args.folder_path, filename_prefix=args.prefix_1,
                            pretrained_emb=args.pretrain, cuda=bool(args.gpu_id >= 0))
    gensen_2 = GenSenSingle(model_folder=args.folder_path, filename_prefix=args.prefix_2,
                            pretrained_emb=args.pretrain, cuda=bool(args.gpu_id >= 0))
    gensen = GenSen(gensen_1, gensen_2)
    global STRATEGY
    STRATEGY = args.strategy
    params_senteval['gensen'] = gensen

    # Do SentEval stuff
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    tasks = get_tasks(args.tasks)
    results = se.eval(tasks)
    print(results)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
