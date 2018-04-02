# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import os
import sys
import torch
from exutil import dotdict
import logging
import argparse


# Set PATHs
GLOVE_PATH = 'glove/glove.840B.300d.txt'
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
MODEL_PATH = 'infersent.allnli.pickle'

assert os.path.isfile(MODEL_PATH) and os.path.isfile(GLOVE_PATH), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples],
                                 tokenize=False)


def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size,
                                         tokenize=False)
    return embeddings

def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--use_pytorch", help="1 to use PyTorch", type=int,
            default=1)
    parser.add_argument("--log_file", help="File to log to", type=str)
    parser.add_argument("--cuda", help="CUDA id to use", type=int, default=0)
    parser.add_argument("--small", help="Use small training data if available", type=int, default=1)
    parser.add_argument("--lower", help="Lower case data", type=int, default=0)

    args = parser.parse_args(arguments)

    if args.cuda:
        logging.debug("Using GPU %d" % args.cuda)
        torch.cuda.device(args.cuda)

    # Set params for SentEval
    params_senteval = {'task_path': PATH_TO_DATA, 'seed': 1111,
            'usepytorch': args.use_pytorch, 'kfold': 5}
    params_senteval = dotdict(params_senteval)

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', 
                        level=logging.DEBUG)
    fileHandler = logging.FileHandler(args.log_file)
    logging.getLogger().addHandler(fileHandler)

    # Load model
    params_senteval.infersent = torch.load(MODEL_PATH)
    params_senteval.infersent.set_glove_path(GLOVE_PATH)

    # Evaluate
    se = senteval.SentEval(params_senteval, batcher, prepare)
    #transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST', 'TREC',
    #                  'MRPC', 'SICKEntailment', 'SICKRelatedness',
    #                  'STSBenchmark', 'STS14', 'SQuAD', 'Quora']
    transfer_tasks = ['Reasoning']
    results = se.eval(transfer_tasks, small=args.small, lower=args.lower)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
