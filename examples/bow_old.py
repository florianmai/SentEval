# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import argparse

from exutil import dotdict
import data
import torch

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data'
PATH_TO_GLOVE = 'glove/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


"""
Note:

The user has to implement two functions:
    1) "batcher" : transforms a batch of sentences into sentence embeddings.
        i) takes as input a "batch", and "params".
        ii) outputs a numpy array of sentence embeddings
        iii) Your sentence encoder should be in "params"
    2) "prepare" : sees the whole dataset, and can create a vocabulary
        i) outputs of "prepare" are stored in "params" that batcher will use.
"""


# consider the option of lower-casing or not for bow.
def prepare(params, samples):
    _, params.word2id = data.create_dictionary(samples)
    params.word_vec = data.get_wordvec(PATH_TO_GLOVE, params.word2id)
    params.wvec_dim = 300
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--use_pytorch", help="1 to use PyTorch", type=int,
                        default=1)
    parser.add_argument("--log_file", help="File to log to", type=str)
    parser.add_argument("--small", help="Use small training data if available",
                        type=int, default=1)
    parser.add_argument("--lower", help="Lower case data", type=int, default=0)

    args = parser.parse_args(arguments)

    # Set params for SentEval
    params_senteval = {'task_path': PATH_TO_DATA,
            'usepytorch': args.use_pytorch, 'kfold': 5}
    params_senteval = dotdict(params_senteval)

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s',
                        level=logging.DEBUG)
    fileHandler = logging.FileHandler(args.log_file)
    logging.getLogger().addHandler(fileHandler)

    se = senteval.SentEval(params_senteval, batcher, prepare)
    #transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST', 'TREC',
    #                  'MRPC', 'SICKEntailment', 'SICKRelatedness',
    #                  'STSBenchmark', 'STS14', 'SQuAD', 'Quora']
    transfer_tasks = ['STS14']
    results = se.eval(transfer_tasks, small=args.small, lower=args.lower)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
