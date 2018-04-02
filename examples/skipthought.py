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


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 512}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load SkipThought model
    #params_senteval.encoder = skipthoughts.load_model()
    params_senteval['encoder'] = skipthoughts.load_model()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SNLI']
    results = se.eval(transfer_tasks)
    print(results)
