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
import cPickle as pkl
import numpy as np

from senteval.tools.validation import SplitClassifier
from senteval.tools.utils import process_sentence, load_tsv, sort_split

class WarstadtEval(object):
    def __init__(self, taskpath, max_seq_len, load_data, seed=1111):
        logging.debug('***** Transfer task : Warstadt Acceptability*****\n\n')
        self.seed = seed
        train = sort_split(self.loadFile(os.path.join(taskpath, 'acceptability_train.tsv'),
                                         max_seq_len, load_data))
        valid = sort_split(self.loadFile(os.path.join(taskpath, 'acceptability_valid.tsv'),
                                         max_seq_len, load_data))
        test = sort_split(self.loadFile(os.path.join(taskpath, 'acceptability_test.tsv'),
                                        max_seq_len, load_data))

        self.samples = train[0] + valid[0] + test[0]
        self.data = {'train': train, 'valid': valid, 'test': test}

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath, max_seq_len, load_data):
        '''Process the dataset located at path.'''
        if os.path.exists(fpath + '.pkl') and load_data:
            data = pkl.load(open(fpath + '.pkl', 'rb'))
            logging.info("Loaded data from %s", fpath + '.pkl')
        else:
            data = load_tsv(fpath, max_seq_len, s1_idx=3, s2_idx=None, targ_idx=1)
            pkl.dump(data, open(fpath + '.pkl', 'wb'))
            logging.info("Saved data to %s", fpath + '.pkl')
        return data


    def run(self, params, batcher):
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1, mylabels = self.data[key]
            enc_input = []
            n_labels = len(mylabels)
            for ii in range(0, n_labels, params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]

                if len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc_input.append(enc1)
                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = mylabels

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for Warstadt Acceptability Judgements\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
