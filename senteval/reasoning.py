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
import numpy as np

from senteval.tools.ranking_validation import SplitClassifier


class ReasoningEval(object):
    def __init__(self, taskpath, seed=1111, lower=0):
        logging.debug('***** Transfer task : Argument Reasoning*****\n\n')
        self.seed = seed
        self.train_rank = train_rank = 1
        self.extra_info = 1
        self.lower = lower

        '''
        split1, split2 respectively contain a context sentence and the question
        split_n_context_sents is a list of # of context sents per context
        split_context_labels is the index of the answer-containing context sent
        '''
        tr_contexts, tr_wars, tr_targs, _, _ = \
            self.loadFile(os.path.join(taskpath, 'reasoning.train'), 
                          ranking=train_rank)
        val_contexts, val_wars, val_targs, val_context_info, val_war_info = \
            self.loadFile(os.path.join(taskpath, 'reasoning.dev'), ranking=0)
        te_contexts, te_wars, te_targs, te_context_info, te_war_info = \
            self.loadFile(os.path.join(taskpath, 'reasoning.test'), ranking=0)
        logging.debug('\tFinished loading data')

        # duplicate samples?
        self.samples = [s for c in tr_contexts + val_contexts + te_contexts + \
                        tr_wars + val_wars + te_wars for s in c]
        self.data = {'train': (tr_contexts, tr_wars, tr_targs),
                     'valid': (val_contexts, val_wars, val_targs),
                     'test': (te_contexts, te_wars, te_targs)}

        # Needed for ranking evaluation
        self.val_context_info = val_context_info # (qID, correct warrant #)
        self.te_context_info = te_context_info
        self.val_war_info = val_war_info # (qID, warrant #)
        self.te_war_info = te_war_info

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath, ranking=0):
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

        with open(fpath) as fh:
            raw_data = fh.readlines()
        data = {}
        for raw_datum in raw_data[1:]:
            arg_id, war0, war1, targ, reason, claim, title, info = \
                    raw_datum.split('\t') # option to use title, info
            context = [claim, reason] # should split into different sents
            if self.extra_info:
                context += [title, info]
            context = [word_tokenize(c) for c in context]
            war0 = word_tokenize(war0)
            war1 = word_tokenize(war1)
            data[arg_id] = (context, war0, war1, int(targ))

        if ranking:
            contexts, war0s, war1s, targs = self.create_ranking_data(data)
            len_sorted = sorted(zip(war0s, war1s, contexts, targs), 
                                  key=lambda z: (len(z[0]), len(z[1])))
            war0s, war1s, contexts, targs = map(list, zip(*len_sorted))
            reshape_contexts = []
            for i in xrange(len(contexts[0])):
                reshape_contexts.append([c[i] for c in contexts])
            wars = [war0s, war1s]
            return reshape_contexts, wars, targs, None, None
        else:
            contexts, wars, targs, context_info, war_info = \
                self.create_binary_data(data)
            len_sorted = sorted(zip(wars, contexts, targs, war_info), 
                                  key=lambda z: (len(z[0])))
            wars, contexts, targs, war_info  = map(list, zip(*len_sorted))
            reshape_contexts = []
            for i in xrange(len(contexts[0])):
                reshape_contexts.append([c[i] for c in contexts])
            wars = [wars]
            return reshape_contexts, wars, targs, context_info, war_info


    def create_ranking_data(self, data):
        '''
        Given sentences and questions, process data into ranking format:
            q1 s11 s12 l
            q1 s11 s13 1 # possibly permute s1, s3, 0
            ...
            q2 s21 s22 1
            ...
            qn sn1 snm 1 # where there are n questions, max m sents / quest
        '''
        contexts, war0s, war1s, targs = [], [], [], []
        for arg_id, (context, war0, war1, targ) in data.iteritems():
            contexts.append(context)
            war0s.append(war0)
            war1s.append(war1)
            if targ: # targ == 1
                targs.append(-1)
            else: # targ == 0
                targs.append(1)
        return contexts, war0s, war1s, targs

    def create_binary_data(self, data):
        '''
        Given sentences and questions, process data into
            binary classification format:

        Returns:
            all_questions: list of questions w/ (# context sents) reps per
            all_contexts: list of all contexts
            all_labels: list of 0/1 labels if corresponding context sent
                contains answering span
            quest_info: list of [(question ID, # context sents, gold_idx)]
            sent_info: list of [(question ID, order in original sentence)]
        '''
        context_info, contexts, wars, targs, war_info = [], [], [], [], []

        for arg_id, (context, war0, war1, targ) in data.iteritems():
            context_info.append((arg_id, targ))
            contexts += [context] * 2
            wars += [war0, war1]
            targs += [int(targ == 0), int(targ == 1)]
            war_info += [(arg_id, 0), (arg_id, 1)]
        assert len(contexts) == len(wars) == len(war_info) == \
                len(targs) == 2*len(context_info)
        assert sum(targs) == len(contexts) / 2
        return contexts, wars, targs, context_info, war_info

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
            contexts, wars, mylabels = all_data
            enc_input = []
            n_labels = len(mylabels)
            for ii in range(0, n_labels, params.batch_size):
                context_encs = [] # variable number of sentences per datum
                for sents in contexts:
                    batch = sents[ii:ii + params.batch_size]
                    enc = batcher(params, batch)
                    context_encs.append(enc)
                war_encs = []
                for sents in wars:
                    batch = sents[ii:ii + params.batch_size]
                    enc = batcher(params, batch)
                    war_encs.append(enc)
                if self.train_rank and key == 'train':
                    prods0 = [war_encs[0] * context_enc for context_enc in context_encs]
                    prods1 = [war_encs[1] * context_enc for context_enc in context_encs]
                    enc_datum = np.hstack(tuple(enc for enc in \
                                [war_encs[0]] + context_encs + prods0 + \
                                [war_encs[1]] + context_encs + prods1))
                else:
                    prods = [war_encs[0] * context_enc for context_enc in context_encs]
                    enc_datum = np.hstack(tuple(enc for enc in \
                                war_encs + context_encs + prods))
                enc_input.append(enc_datum)
                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = mylabels

        '''
        Create classifier that does train, validation, test
        '''

        train_rank = self.train_rank
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
                try:
                    assert val_rankings[cur_q+i][0] == arg_id
                except Exception as e:
                    pdb.set_trace()
            cur_q += 2
        cur_q = 0
        for q in self.te_context_info:
            arg_id, _ = q[0], q[1]
            for i in xrange(2):
                assert te_rankings[cur_q+i][0] == arg_id
            cur_q += 2

        logging.info('Dev binary acc : {0} Test acc : {1} for Reasoning\n'
                      .format(devacc, testacc))
        logging.info('Dev acc: {0} for argument reasoning\n'
                      .format(rank(self.val_context_info, val_rankings)))
        logging.info('Test acc: {0} for argument reasoning\n'
                      .format(rank(self.te_context_info, te_rankings)))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
