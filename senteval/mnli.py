# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SNLI - Entailment
'''
from __future__ import absolute_import, division, unicode_literals

import os
import copy
import codecs
import logging
import numpy as np

from senteval.tools.validation import SplitClassifier
from senteval.tools.utils import process_sentence, load_tsv, sort_split

class MNLIEval(object):
    def __init__(self, taskpath, max_seq_len=50, seed=1111):
        logging.debug('***** Transfer task : MNLI Entailment*****\n\n')
        self.seed = seed
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        #train1, train2, trainlabels = self.loadFile(os.path.join(taskpath, 'multinli_1.0_train.txt'), targ_map)
        #valid1, valid2, validlabels = self.loadFile(os.path.join(taskpath, 'multinli_1.0_dev_matched.txt'))
        #test1, test2, testlabels = self.loadFile(os.path.join(taskpath, 'multinli_1.0_dev_matched.txt'))
        #test1, test2, testlabels = self.loadAux(os.path.join(taskpath, 'adversarial_nli.tsv'))

        train = sort_split(self.loadFile(os.path.join(taskpath, 'multinli_1.0_train.txt'),
                           max_seq_len, targ_map))
        valid = sort_split(self.loadFile(os.path.join(taskpath, 'multinli_1.0_dev_matched.txt'),
                           max_seq_len, targ_map))
        test = sort_split(self.loadFile(os.path.join(taskpath, 'multinli_1.0_dev_matched.txt'),
                          max_seq_len, targ_map))

        # sort data (by s2 first) to reduce padding
        '''
        sorted_train = sorted(zip(train2, train1, trainlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))
        train2, train1, trainlabels = map(list, zip(*sorted_train))

        sorted_valid = sorted(zip(valid2, valid1, validlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))
        valid2, valid1, validlabels = map(list, zip(*sorted_valid))

        sorted_test = sorted(zip(test2, test1, testlabels),
                             key=lambda z: (len(z[0]), len(z[1]), z[2]))
        test2, test1, testlabels = map(list, zip(*sorted_test))
        self.samples = train1 + train2 + valid1 + valid2 + test1 + test2
        self.data = {'train': (train1, train2, trainlabels),
                     'valid': (valid1, valid2, validlabels),
                     'test': (test1, test2, testlabels)}
        '''

        self.samples = train[0] + train[1] + valid[0] + valid[1] + test[0] + test[1]
        self.data = {'train': train, 'valid': valid, 'test': test}

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath, max_seq_len, targ_map):
        '''Process the dataset located at path.'''
        return load_tsv(fpath, max_seq_len, s1_idx=5, s2_idx=6, targ_idx=0,
                        targ_map=targ_map, skip_rows=1)

    def loadAux(self, fpath, max_seq_len, targ_map):
        '''Process the dataset located at path.'''
        return load_tsv(fpath, max_seq_len, s1_idx=6, s2_idx=7, targ_idx=8, targ_map=targ_map,
                        skip_rows=1)

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
                    logging.info("PROGRESS (encoding): %.2f%%" % (100 * ii / n_labels))
            logging.debug("Finished encoding MNLI")
            self.X[key] = np.vstack(enc_input)
            self.y[key] = mylabels

        config = {'nclasses': 3, 'seed': self.seed, 'usepytorch': params.usepytorch,
                  'cudaEfficient': True, 'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        logging.debug("Built classifier, starting training")
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for MNLI\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
