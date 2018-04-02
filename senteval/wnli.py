# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
WNLI
'''
from __future__ import absolute_import, division, unicode_literals

import os
import pdb
import copy
import codecs
import logging
import numpy as np

from senteval.tools.validation import SplitClassifier
from senteval.tools.utils import process_sentence, load_tsv, sort_split

class WNLIEval(object):
    def __init__(self, taskpath, max_seq_len, seed=1111):
        logging.debug('***** Transfer task : WNLI Entailment*****\n\n')
        self.seed = seed
        train = sort_split(self.loadFile(os.path.join(taskpath, 'wnli_train.tsv'), max_seq_len))
        valid = sort_split(self.loadFile(os.path.join(taskpath, 'wnli_valid.tsv'), max_seq_len))
        test = sort_split(self.loadFile(os.path.join(taskpath, 'wnli_test.tsv'), max_seq_len))

        self.samples = train[0] + train[1] + valid[0] + valid[1] + test[0] + test[1]
        self.data = {'train': train, 'valid': valid, 'test': test}

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath, max_seq_len):
        '''Process the dataset located at path.'''
        return load_tsv(fpath, max_seq_len)

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1, input2, mylabels = self.data[key]
            enc_input = []
            n_labels = len(mylabels)
            for ii in range(0, n_labels, params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                np.abs(enc1 - enc2))))
                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = mylabels

        config = {'nclasses': 3, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for WNLI\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
