'''
Quora Question Pairs
'''
from __future__ import absolute_import, division, unicode_literals

import os
import pdb
import logging
import nltk
import numpy as np
from sklearn.metrics import f1_score

from senteval.tools.validation import KFoldClassifier
from senteval.tools.utils import process_sentence, load_tsv, sort_split


class QuoraEval(object):
    def __init__(self, taskpath, max_seq_len, seed=1111):
        logging.debug('***** Transfer task : Quora Question Similarity*****\n\n')
        self.seed = seed
        train = sort_split(self.loadFile(os.path.join(taskpath, 'quora_duplicate_questions_clean.tsv'), max_seq_len))
        test = sort_split(self.loadFile(os.path.join(taskpath, 'quora_test.tsv'), max_seq_len))

        self.samples = train[0] + train[1] + test[0] + test[1]
        self.data = {'train': train, 'test': test}

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath, max_seq_len):
        '''
        Read in and process data directly from JSON
        Returns dictionary with format
            question_ID: (question, tokenized contexts, 0/1 labels, gold_idx)
        '''
        return load_tsv(fpath, max_seq_len, s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)

    def run(self, params, batcher):
        embed = {'train': {}, 'test': {}}

        for key in self.data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            sorted_corpus = sorted(zip(self.data[key][0], self.data[key][1], self.data[key][2]),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            text_data['A'] = [x for (x, y, z) in sorted_corpus]
            text_data['B'] = [y for (x, y, z) in sorted_corpus]
            text_data['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['A', 'B']:
                embed[key][txt_type] = []
                for ii in range(0, len(text_data['y']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    embed[key][txt_type].append(embeddings)
                embed[key][txt_type] = np.vstack(embed[key][txt_type])
            embed[key]['y'] = np.array(text_data['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = embed['train']['A']
        trainB = embed['train']['B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = embed['train']['y']

        # Test
        testA = embed['test']['A']
        testB = embed['test']['B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = embed['test']['y']

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = KFoldClassifier(train={'X': trainF, 'y': trainY},
                              test={'X': testF, 'y': testY}, config=config)

        devacc, testacc, yhat = clf.run()
        testf1 = round(100*f1_score(testY, yhat), 2)
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for Quora.\n'
                      .format(devacc, testacc, testf1))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1,
                'ndev': len(trainA), 'ntest': len(testA)}
