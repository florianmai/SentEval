# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

"""
Example of file to compare skipthought vectors with our InferSent model
"""
import logging
import argparse
from exutil import dotdict
import sys


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_SKIPTHOUGHT = '/misc/vlgscratch4/BowmanGroup/awang/models/skip-thoughts'
PATH_TO_TOOLS = '/misc/vlgscratch4/BowmanGroup/awang/models/skip-thoughts/training'
assert PATH_TO_SKIPTHOUGHT != '', 'Download skipthought and set correct PATH'

# import skipthought and Senteval
sys.path.insert(0, PATH_TO_SKIPTHOUGHT)
sys.path.insert(0, PATH_TO_TOOLS)
sys.path.insert(0, PATH_TO_SENTEVAL)
import skipthoughts
import tools
import senteval


def prepare(params, samples):
    return

def batcher(params, batch):
    embeddings = tools.encode(params.encoder,
    #embeddings = skipthoughts.encode(params.encoder,
                                     [' '.join(sent).strip()
                                      if sent != [] else '.' for sent in batch],
                                     verbose=False, use_eos=True)
    return embeddings

def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--use_pytorch", help="1 to use PyTorch", type=int,
            default=1)
    parser.add_argument("--log_file", help="File to log to", type=str)
    parser.add_argument("--dict_file", help="File to load dict from", 
                        type=str)
    parser.add_argument("--model_file", help="File to load model from", 
                        type=str)
    parser.add_argument("--small", help="Use small training data if available", type=int, default=1)
    parser.add_argument("--lower", help="Lower case data", type=int, default=0)

    args = parser.parse_args(arguments)

    # Set params for SentEval
    params_senteval = {'usepytorch': True,
                       'task_path': PATH_TO_DATA,
                       'batch_size': 512}
    params_senteval = dotdict(params_senteval)

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    fileHandler = logging.FileHandler(args.log_file)
    logging.getLogger().addHandler(fileHandler)

    embed_map = tools.load_googlenews_vectors()
    params_senteval.encoder = tools.load_model(embed_map,
                                               args.model_file,
                                               args.dict_file)
    #params_senteval.encoder = skipthoughts.load_model()
    se = senteval.SentEval(params_senteval, batcher, prepare)
    tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 
             'TREC', 'SICKRelatedness','SICKEntailment', 
             'MRPC', 'STS14', 'SQuAD', 'Quora', 'Reasoning']
    #tasks = ['Quora', 'Reasoning']

    se.eval(tasks, small=args.small, lower=args.lower)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
