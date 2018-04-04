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
import logging
import pdb
import sys
import argparse
#sys.setdefaultencoding('utf8')


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_SKIPTHOUGHT = '/misc/vlgscratch4/BowmanGroup/awang/models/skip-thoughts/'

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
    parser.add_argument("--cuda", help="CUDA id to use", type=int, default=0)
    parser.add_argument("--use_pytorch", help="1 to use PyTorch", type=int, default=1)
    parser.add_argument("--log_file", help="File to log to", type=str)
    parser.add_argument("--dict_file", help="File to load dict from", type=str)
    parser.add_argument("--model_file", help="File to load model from", type=str)

    # Task options
    parser.add_argument("--tasks", help="Tasks to evaluate on, as a comma separated list", type=str)
    parser.add_argument("--max_seq_len", help="Max sequence length", type=int, default=100)

    # Model options
    parser.add_argument("--batch_size", help="Batch size to use", type=int, default=32)

    # Classifier options
    parser.add_argument("--cls_batch_size", help="Batch size to use for classifiers", type=int, default=32)

    args = parser.parse_args(arguments)
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    # Set params for SentEval
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10,
                       'max_seq_len': args.max_seq_len, 'batch_size': args.batch_size}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': args.cls_batch_size,
                                     'tenacity': 5, 'epoch_size': 4}
    params_senteval['encoder'] = skipthoughts.load_model()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    tasks = args.tasks.split(',')
    results = se.eval(tasks)
    print(results)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
