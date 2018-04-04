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

import data

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data'
PATH_TO_GLOVE = 'glove/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


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

    # Logistics
    parser.add_argument("--cuda", help="CUDA id to use", type=int, default=0)
    parser.add_argument("--use_pytorch", help="1 to use PyTorch", type=int, default=1)
    parser.add_argument("--log_file", help="File to log to", type=str)

    # Task options
    parser.add_argument("--tasks", help="Tasks to evaluate on, as a comma separated list", type=str)
    parser.add_argument("--max_seq_len", help="Max sequence length", type=int, default=100)

    # Model options
    parser.add_argument("--batch_size", help="Batch size to use", type=int, default=32)

    # Classifier options
    parser.add_argument("--cls_batch_size", help="Batch size to use for classifier", type=int, default=32)

    args = parser.parse_args(arguments)
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    fileHandler = logging.FileHandler(args.log_file)
    logging.getLogger().addHandler(fileHandler)

    # SentEval params
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10,
                       'max_seq_len': args.max_seq_len, 'batch_size': args.batch_size}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': args.cls_batch_size,
                                     'tenacity': 5, 'epoch_size': 4}

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    tasks = args.tasks.split(',')
    results = se.eval(tasks)
    print(results)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
