# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
MRPC : Microsoft Research Paraphrase (detection) Corpus
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import logging
import ipdb as pdb
import cPickle as pkl
import numpy as np
from sklearn.metrics import f1_score

from senteval.tools.validation import KFoldClassifier
from senteval.tools.utils import process_sentence, load_tsv, load_test, sort_split


class MRPCEval(object):
    def __init__(self, task_path, max_seq_len, load_data, seed=1111):
        logging.info('***** Transfer task : MRPC *****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'msr_paraphrase_train.txt'), max_seq_len, load_data)
        #test = self.loadFile(os.path.join(task_path, 'msr_paraphrase_test.txt'), max_seq_len, load_data)
        #self.samples = train['X_A'] + train['X_B'] + test['X_A'] + test['X_B']
        test = self.loadTest(os.path.join(task_path, 'msrp_test_ans.tsv'), max_seq_len, load_data)
        self.samples = train['X_A'] + train['X_B'] + test[0] + test[1]
        self.mrpc_data = {'train': train, 'test': test}

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        return prepare(params, self.samples)

    def loadTest(self, data_file, max_seq_len, load_data):
        '''Load indexed data'''
        if os.path.exists(data_file + '.pkl') and load_data:
            data = pkl.load(open(data_file + '.pkl', 'rb'))
            logging.info("Loaded data from %s", data_file + '.pkl')
        else:
            data = load_test(data_file, max_seq_len, s1_idx=1, s2_idx=2, targ_idx=3, idx_idx=0,
                             skip_rows=1)
            pkl.dump(data, open(data_file + '.pkl', 'wb'))
            logging.info("Saved data to %s", data_file + '.pkl')
        return data

    def loadFile(self, fpath, max_seq_len, load_data):
        if os.path.exists(fpath + '.pkl') and load_data:
            mrpc_data = pkl.load(open(fpath + '.pkl', 'rb'))
            logging.info("Loaded data from %s", fpath + '.pkl')
        else:
            mrpc_data = {'X_A': [], 'X_B': [], 'y': []}
            with io.open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    text = line.strip().split('\t')
                    mrpc_data['X_A'].append(process_sentence(text[3], max_seq_len))
                    mrpc_data['X_B'].append(process_sentence(text[4], max_seq_len))
                    mrpc_data['y'].append(text[0])

            mrpc_data['X_A'] = mrpc_data['X_A'][1:]
            mrpc_data['X_B'] = mrpc_data['X_B'][1:]
            mrpc_data['y'] = [int(s) for s in mrpc_data['y'][1:]]
            pkl.dump(mrpc_data, open(fpath + '.pkl', 'wb'))
            logging.info("Saved data to %s", fpath + '.pkl')
        return mrpc_data

    def run(self, params, batcher):
        mrpc_embed = {'train': {}, 'test': {}}

        for key in self.mrpc_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            if key == 'train':
                sorted_corpus = sorted(zip(self.mrpc_data[key]['X_A'],
                                           self.mrpc_data[key]['X_B'],
                                           self.mrpc_data[key]['y']),
                                       key=lambda z: (len(z[0]), len(z[1]), z[2]))
                text_data['A'] = [x for (x, y, z) in sorted_corpus]
                text_data['B'] = [y for (x, y, z) in sorted_corpus]
                text_data['y'] = [z for (x, y, z) in sorted_corpus]
            else:
                sorted_corpus = sorted(zip(self.mrpc_data[key][0],
                                           self.mrpc_data[key][1],
                                           self.mrpc_data[key][2],
                                           self.mrpc_data[key][3]),
                                       key=lambda z: (len(z[0]), len(z[1]), z[2]))
                text_data['A'] = [x for (x, y, z, w) in sorted_corpus]
                text_data['B'] = [y for (x, y, z, w) in sorted_corpus]
                text_data['y'] = [z for (x, y, z, w) in sorted_corpus]
                text_data['idx'] = [w for (x, y, z, w) in sorted_corpus]
                mrpc_embed[key]['idx'] = text_data['idx']

            for txt_type in ['A', 'B']:
                mrpc_embed[key][txt_type] = []
                for ii in range(0, len(text_data['y']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    mrpc_embed[key][txt_type].append(embeddings)
                mrpc_embed[key][txt_type] = np.vstack(mrpc_embed[key][txt_type])
            mrpc_embed[key]['y'] = np.array(text_data['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = mrpc_embed['train']['A']
        trainB = mrpc_embed['train']['B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = mrpc_embed['train']['y']

        # Test
        testA = mrpc_embed['test']['A']
        testB = mrpc_embed['test']['B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = mrpc_embed['test']['y']
        testIdxs = mrpc_embed['test']['idx']

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = KFoldClassifier(train={'X': trainF, 'y': trainY},
                              test={'X': testF, 'y': testY}, config=config)
        devacc, testacc, yhat = clf.run()
        testf1 = round(100*f1_score(testY, yhat), 2)
        idxs_and_preds = [(idx, pred) for idx, pred in zip(testIdxs, yhat.squeeze().tolist())]
        idxs_and_preds.sort(key=lambda x: x[0])
        preds = [pred for _, pred in idxs_and_preds]
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for MRPC.\n'
                      .format(devacc, testacc, testf1))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1, 'preds': preds,
                'ndev': len(trainA), 'ntest': len(testA)}
