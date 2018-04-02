'''
SQuAD - Ranking
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import pdb
import json
import nltk
import logging
import xml.etree.ElementTree as ET
import numpy as np

from senteval.tools.ranking_validation import SplitClassifier


class RTE8Eval(object):
    def __init__(self, taskpath, seed=1111, lower=0, granularity=2):
        logging.debug('***** Transfer task : RTE8 *****\n\n')
        self.seed = seed
        self.train_binary = train_binary = 0 # 1 if use BCE to train, 0 for margin ranking
        self.extra_info = 0
        self.lower = lower
        assert granularity in [2,3,5]
        self.granularity = granularity
        if granularity == 2: # or I could just rearrange the data...
            data_dir = '2way'
        elif granulrity == 3:
            data_dir = '3way'

        '''
        split1, split2 respectively contain a context sentence and the question
        split_n_context_sents is a list of # of context sents per context
        split_context_labels is the index of the answer-containing context sent
        '''
        tr_quests, tr_anss, tr_targs = \
                self.loadFile(os.path.join(taskpath, 
                              'training/%s/sciEntsBank' % data_dir))
        val_quests, val_anss, val_targs = \
                self.loadFile(os.path.join(taskpath, 
                              'dev/%s/sciEntsBank' % data_dir))
        te_quests, te_anss, te_targs = \
                self.loadFile(os.path.join(taskpath, 
                              'test/%s/sciEntsBank' % data_dir))
        val_quests, val_anss, val_targs = \
                te_quests[:500], te_anss[:500], te_targs[:500]
        te_quests, te_anss, te_targs = \
                te_quests[500:], te_anss[500:], te_targs[500:]

        logging.debug('\tFinished loading data')

        # sort data by context length then question length to reduce padding
        sorted_train = sorted(zip(tr_quests, tr_anss, tr_targs), 
                                key=lambda z: (len(z[1]), len(z[0])))
        tr_quests, tr_anss, tr_targs = map(list, zip(*sorted_train))
        sorted_valid = sorted(zip(val_quests, val_anss, val_targs), 
                                key=lambda z: (len(z[1]), len(z[0])))
        val_quests, val_anss, val_targs = map(list, zip(*sorted_valid))
        sorted_test = sorted(zip(te_quests, te_anss, te_targs), 
                                key=lambda z: (len(z[1]), len(z[0])))
        te_quests, te_anss, te_targs = map(list, zip(*sorted_test))
        logging.debug('\tFinished sorting data')

        self.samples = tr_quests + tr_anss + val_quests + val_anss + \
                        te_quests + te_anss
        self.data = {'train': (tr_quests, tr_anss, tr_targs),
                     'valid': (val_quests, val_anss, val_targs),
                     'test': (te_quests, te_anss, te_targs)}

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        '''
        Read in and process data directly from JSON
        Returns dictionary with format
            question_ID: (question, tokenized contexts, 0/1 labels, gold_idx)
        '''
        def word_tokenize(sent):
            toks = nltk.word_tokenize(sent.strip())
            if self.lower:
                toks = [w.strip().lower() for w in toks]
            return toks

        quests, anss, targs = [], [], []
        for quest_file in os.listdir(fpath):
            if 'xml' not in quest_file:
                continue

            tree = ET.parse(fpath + '/' + quest_file)
            quest = tree.getroot()
            quest_text = word_tokenize(quest.findall('questionText')[0].text)

            ref_anss = quest.findall('referenceAnswers')[0]

            stu_anss = quest.findall('studentAnswers')[0]
            for stu_ans in stu_anss:
                anss.append(word_tokenize(stu_ans.text))
                quests.append(quest_text)
                targs.append(int(stu_ans.attrib['accuracy'] == 'correct'))

        return quests, anss, targs

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        for key in self.data:
            '''
            For each split, encode sentences as features
            '''
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            all_data = self.data[key]
            if len(all_data) == 3:
                input1, input2, mylabels = all_data
                input3 = None
            else:
                assert len(all_data) == 4
                input1, input2, input3, mylabels = all_data
            enc_input = []
            n_labels = len(mylabels)
            #fracs, n_zeros, n_ins = [], 0, 0.
            for ii in range(0, n_labels, params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]
                if input3 is not None:
                    batch3 = input3[ii:ii + params.batch_size]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    #enc1, fracs1, n_zeros1 = batcher(params, batch1)
                    #enc2, fracs2, n_zeros2 = batcher(params, batch2)
                    #n_zeros += n_zeros1 + n_zeros2
                    #fracs += [fracs1, fracs2]
                    #n_ins += len(batch1) + len(batch2)
                    if input3 is not None:
                        enc3 = batcher(params, batch3)
                        #enc3, fracs3, n_zeros3 = batcher(params, batch3)
                        #fracs.append(fracs3)
                        #n_zeros += n_zeros3
                        #n_ins += len(batch3)
                        enc_datum = np.hstack((enc1, enc2, enc1 * enc2, 
                            np.abs(enc1 - enc2), enc1, enc3, enc1 * enc3, 
                            np.abs(enc1 - enc3)))
                    else:
                        enc_datum = np.hstack((enc1, enc2, enc1 * enc2,
                                               np.abs(enc1 - enc2)))
                    enc_input.append(enc_datum)
                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            #logging.debug("Split: %s" % key)
            #logging.debug("\tmean frac in vocab: %.3f, mean n_zeros: %.3f, n_ins: %d" % (sum(fracs) / n_ins, n_zeros / n_ins, n_ins))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = mylabels

        '''
        Create classifier that does train, validation, test
        '''

        train_rank = not self.train_binary
        config_classifier = {'nclasses': 2, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'cudaEfficient': True,
                             'classifier': params.classifier,
                             'nhid': params.nhid, 'maxepoch': 15,
                             'nepoches': 1, 'noreg': False,
                             'train_rank': train_rank}
        clf = SplitClassifier(self.X, self.y, config_classifier)
        devacc, testacc, devprobs, testprobs = clf.run()

        def rank(context_info, rankings):
            n_correct = 0.
            n_contexts = len(context_info)
            for i in xrange(n_contexts):
                arg_id, targ = context_info[i]
                if targ and rankings[2*i][2][1] < rankings[2*i+1][2][1]:
                    n_correct += 1
                if not targ and rankings[2*i][2][1] > rankings[2*i+1][2][1]:
                    n_correct += 1
            return round(100*n_correct / n_contexts, 2)

        # group by question, then probability
        val_rankings = [(arg_id, war_id, prob) for (arg_id, war_id), prob in \
                zip(self.val_war_info, devprobs)]
        val_rankings.sort(key=lambda z: (z[0], -z[1]))
        te_rankings = [(arg_id, war_id, prob) for (arg_id, war_id), prob in \
                zip(self.te_war_info, testprobs)]
        te_rankings.sort(key=lambda z: (z[0], -z[1]))
        self.val_context_info.sort(key=lambda z: z[0])
        self.te_context_info.sort(key=lambda z: z[0])

        # check that I'm not an idiot
        cur_q = 0
        for q in self.val_context_info:
            arg_id, _ = q[0], q[1]
            for i in xrange(2):
                assert val_rankings[cur_q+i][0] == arg_id
            cur_q += 2
        cur_q = 0
        for q in self.te_context_info:
            arg_id, _ = q[0], q[1]
            for i in xrange(2):
                assert te_rankings[cur_q+i][0] == arg_id
            cur_q += 2

        logging.info('Dev binary acc : {0} Test acc : {1} for RTE8\n'
                      .format(devacc, testacc))
        logging.info('Dev acc: {0} for RTE8\n'
                      .format(rank(self.val_context_info, val_rankings)))
        logging.info('Test acc: {0} for RTE8\n'
                      .format(rank(self.te_context_info, te_rankings)))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
