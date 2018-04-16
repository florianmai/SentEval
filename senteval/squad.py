# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
(Adversarial) SQuAD - binary
'''
from __future__ import absolute_import, division, unicode_literals

import os
import pdb
import json
import copy
import logging
import cPickle as pkl
import numpy as np

from senteval.tools.validation import SplitClassifier
from senteval.tools.utils import process_sentence, load_tsv, sort_split, load_test, sort_preds


class SQuADEval(object):
    def __init__(self, taskpath, max_seq_len, load_data, seed=1111):
        logging.debug('***** Transfer task : SQuAD Classification *****\n\n')
        self.seed = seed

        train = sort_split(self.loadFile(os.path.join(taskpath, "adv_squad_train.json"), max_seq_len, load_data))
        valid = sort_split(self.loadFile(os.path.join(taskpath, "adv_squad_dev.json"), max_seq_len, load_data))
        #test = sort_split(self.loadFile(os.path.join(taskpath, "adv_squad_test.json"), max_seq_len, load_data))
        test = sort_split(self.loadTest(os.path.join(taskpath, "squad_test_ans.tsv"), max_seq_len, load_data))

        # sort data (by s2 first) to reduce padding
        self.samples = train[0] + train[1] + valid[0] + valid[1] + test[0] + test[1]
        self.data = {'train': train, 'valid': valid, 'test': test}

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath, max_seq_len, load_data):
        '''Load a single split'''
        if os.path.exists(fpath + '.pkl') and load_data:
            quests, ctxs, targs = pkl.load(open(fpath + '.pkl', 'rb'))
            logging.info("Loaded data from %s", fpath + '.pkl')
        else:
            quests, ctxs, targs = [], [], []
            data = json.load(open(fpath))
            for datum in data:
                quests.append(process_sentence(datum['question'], max_seq_len))
                ctxs.append(process_sentence(datum['sentence'], max_seq_len))
                assert datum['label'] in ['True', 'False'], pdb.set_trace()
                targs.append(int(datum['label'] == 'True'))
            pkl.dump((quests, ctxs, targs), open(fpath + '.pkl', 'wb'))
            logging.info("Saved data to %s", fpath + '.pkl')
        return quests, ctxs, targs

    def loadTest(self, data_file, max_seq_len, load_data):
        '''Load indexed data'''
        if os.path.exists(data_file + '.pkl') and load_data:
            data = pkl.load(open(data_file + '.pkl', 'rb'))
            logging.info("Loaded data from %s", data_file + '.pkl')
        else:
            targ_map = {'not_contain': 0, 'contains': 1}
            data = load_test(data_file, max_seq_len, s1_idx=1, s2_idx=2, targ_idx=3,
                             idx_idx=0, skip_rows=1, targ_map=targ_map)
            pkl.dump(data, open(data_file + '.pkl', 'wb'))
            logging.info("Saved data to %s", data_file + '.pkl')
        return data

    def run(self, params, batcher):
        self.X, self.y, self.idxs = {}, {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []
            if key not in self.idxs:
                self.idxs[key] = []

            if len(self.data[key]) == 3:
                input1, input2, mylabels = self.data[key]
            else:
                input1, input2, mylabels, idxs = self.data[key]
                self.idxs[key]= idxs
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
            self.X[key] = np.vstack(enc_input)
            self.y[key] = mylabels #[dico_label[y] for y in mylabels]

        config = {'nclasses': 2, 'seed': self.seed, 'usepytorch': params.usepytorch,
                  'cudaEfficient': True, 'nhid': params.nhid, 'noreg': False}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc, test_preds = clf.run()
        test_preds = sort_preds(test_preds.squeeze().tolist(), self.idxs['test'])
        logging.debug('Dev acc : {0} Test acc : {1} for SQuAD\n' .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'preds': test_preds,
                'ndev': len(self.data['valid'][0]), 'ntest': len(self.data['test'][0])}
