'''
Quora Question Pairs
'''
from __future__ import absolute_import, division, unicode_literals

import os
import ipdb as pdb
import copy
import logging
import cPickle as pkl
import numpy as np
from sklearn.metrics import f1_score

from senteval.tools.validation import SplitClassifier #KFoldClassifier
from senteval.tools.utils import process_sentence, load_tsv, sort_split, split_split


class QuoraEval(object):
    def __init__(self, taskpath, max_seq_len, load_data, seed=1111):
        logging.debug('***** Transfer task : Quora Question Similarity*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(taskpath, 'quora_duplicate_questions_clean.tsv'),
                              max_seq_len, load_data)
        train, valid = split_split(train)
        train = sort_split(train)
        valid = sort_split(valid)
        test = sort_split(self.loadTest(os.path.join(taskpath, 'quora_test_ans.tsv'), max_seq_len, load_data))

        self.samples = train[0] + train[1] + valid[0] + valid[1] + test[0] + test[1]
        self.data = {'train': train, 'valid': valid, 'test': test}

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath, max_seq_len, load_data):
        '''
        Read in and process data directly from JSON
        Returns dictionary with format
            question_ID: (question, tokenized contexts, 0/1 labels, gold_idx)
        '''
        if os.path.exists(fpath + '.pkl') and load_data:
            data = pkl.load(open(fpath + '.pkl', 'rb'))
            logging.info("Loaded data from %s", fpath + '.pkl')
        else:
            data = load_tsv(fpath, max_seq_len, s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
            pkl.dump(data, open(fpath + '.pkl', 'wb'))
            logging.info("Saved data to %s", fpath + '.pkl')
        return data

    def loadTest(self, data_file, max_seq_len, load_data):
        if os.path.exists(data_file + '.pkl') and load_data:
            data = pkl.load(open(data_file + '.pkl', 'rb'))
            logging.info("Loaded data from %s", data_file + '.pkl')
        else:
            data = load_tsv(data_file, max_seq_len, s1_idx=2, s2_idx=3, targ_idx=4, skip_rows=1)
            pkl.dump(data, open(data_file + '.pkl', 'wb'))
            logging.info("Saved data to %s", data_file + '.pkl')
        return data

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
                    enc_input.append(np.hstack((enc1, enc2, enc1 * enc2, np.abs(enc1 - enc2))))
                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" % (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = mylabels

        config = {'nclasses': 2, 'seed': self.seed, 'usepytorch': params.usepytorch,
                  'cudaEfficient': True, 'nhid': params.nhid, 'noreg': False}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc, test_preds = clf.run()
        testf1 = round(100*f1_score(self.y['test'], test_preds), 2)
        logging.debug('Dev acc : {0} Test acc : {1} for Quora\n' .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1, 'preds': test_preds,
                'ndev': len(self.data['valid'][0]), 'ntest': len(self.data['test'][0])}
