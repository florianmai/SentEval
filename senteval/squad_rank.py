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


class SQuADEval(object):
    def __init__(self, taskpath, seed=1111, small=0, lower=0):
        logging.debug('***** Transfer task : SQuAD Ranking*****\n\n')
        self.seed = seed
        '''
        split1, split2 respectively contain a context sentence and the question
        split_n_context_sents is a list of # of context sents per context
        split_context_labels is the index of the answer-containing context sent
        '''
        if small:
            tr_file = 'squad.train.small'
        else:
            tr_file = 'squad.train'
        logging.debug("\tLoaded training data from %s" % tr_file)
        tr_data = self.loadFile(os.path.join(taskpath, tr_file), lower=lower)
        val_data = self.loadFile(os.path.join(taskpath, 'squad.dev'), lower=lower)
        te_data = self.loadFile(os.path.join(taskpath, 'squad.test'), lower=lower)
        logging.debug('\tFinished loading data')

        tr_quests, tr_contexts1, tr_contexts2, tr_labels = \
                self.create_ranking_data(tr_data)
        val_quests, val_sents, val_labels, val_quest_info, val_sent_info = \
                self.create_binary_data(val_data)
        te_quests, te_sents, te_labels, te_quest_info, te_sent_info = \
                self.create_binary_data(te_data)
        logging.debug('\tFinished processing data')

        # sort data by context length then question length to reduce padding
        sorted_train = sorted(zip(tr_contexts1, tr_contexts2, tr_quests, tr_labels),
                              key=lambda z: (len(z[0]), len(z[1]), len(z[2])))
        tr_contexts1, tr_contexts2, tr_quests, tr_labels = \
                                                map(list, zip(*sorted_train))

        sorted_valid = sorted(zip(val_sents, val_quests, val_labels, val_sent_info),
                              key=lambda z: (len(z[0]), len(z[1])))
        val_sents, val_quests, val_labels, val_sent_info = \
                                                map(list, zip(*sorted_valid))

        sorted_test = sorted(zip(te_sents, te_quests, te_labels, te_sent_info),
                             key=lambda z: (len(z[0]), len(z[1])))
        te_sents, te_quests, te_labels, te_sent_info = map(list, zip(*sorted_test))
        logging.debug('\tFinished sorting data')

        self.samples = tr_contexts1 + tr_contexts2 + tr_quests + \
                        val_sents + val_quests + te_sents + te_quests
        self.data = {'train': (tr_quests, tr_contexts1, tr_contexts2, tr_labels),
                     'valid': (val_quests, val_sents, val_labels),
                     'test': (te_quests, te_sents, te_labels)
                    }

        # Needed for ranking evaluation
        self.val_quest_info = val_quest_info # (qID, n_sents, gold_sent_idx)
        self.te_quest_info = te_quest_info
        self.val_sent_info = val_sent_info # (qID, order in context)
        self.te_sent_info = te_sent_info

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath, lower):
        '''
        Read in and process data directly from JSON
        Returns dictionary with format
            question_ID: (question, tokenized contexts, 0/1 labels, gold_idx)
        '''
        with open(fpath) as fh:
            raw_data = json.load(fh)
        data, qID = {}, 0
        for i, topic in enumerate(raw_data['data']):
            for j, paragraph in enumerate(topic['paragraphs']):
                context = paragraph['context']
                contexts = nltk.sent_tokenize(context)
                if lower:
                    tok_contexts = [[w.lower() for w in
                       nltk.word_tokenize(c)] for c in contexts]
                else:
                    tok_contexts = [nltk.word_tokenize(c) for c in contexts]
                assert len(contexts) == len(tok_contexts)

                for k, qa in enumerate(paragraph['qas']):
                    if lower:
                        tok_question = [w.lower() for w in 
                                nltk.word_tokenize(qa['question'])]
                    else:
                        tok_question = nltk.word_tokenize(qa['question'])

                    # get the starting location of the first character of span
                    # simple aggregation of Turk labels
                    if len(qa['answers']) == 1:
                        start_idx = qa['answers'][0]['answer_start']
                    elif len(qa['answers']) == 2:
                        start_idx = min(qa['answers'][0]['answer_start'],
                                        qa['answers'][1]['answer_start'])
                    elif qa['answers'][0]['answer_start'] == \
                            qa['answers'][1]['answer_start'] or \
                        qa['answers'][0]['answer_start'] == \
                            qa['answers'][2]['answer_start']:
                        start_idx = qa['answers'][0]['answer_start']
                    else:
                        start_idx = qa['answers'][1]['answer_start']

                    # get the sentence ID of the sentence
                    # containing correct span
                    gold_idx = -1
                    labels = []
                    for c_idx, (c, tokens) in enumerate(zip(contexts, tok_contexts)):
                        # finding sentence index containing span
                        # basically subtract sent lengths from character 
                        # starting point until <= length of current sent 
                        # (i.e. it's in this sentence)
                        c_len = len(c) + 1
                        if start_idx < c_len and start_idx >= 0: 
                            assert gold_idx == -1
                            gold_idx = c_idx
                            labels.append(1)
                        else:
                            labels.append(0)
                        start_idx -= c_len
                    data[qID] = (tok_question, tok_contexts, labels, gold_idx)
                    qID += 1

        return data

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
        all_quests, contexts1, contexts2, all_labels = [], [], [], []

        for qID, (quest, contexts, labels, gold_idx) in data.iteritems():
            n_contexts = len(contexts)
            gold_context = contexts[gold_idx]
            for c_idx, context in enumerate(contexts):
                if c_idx == gold_idx:
                    continue
                contexts1.append(gold_context)
                contexts2.append(context)
                all_labels.append(1)
            all_quests += [quest for _ in range(n_contexts - 1)]

        return all_quests, contexts1, contexts2, all_labels

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
        quest_info, all_quests, all_contexts, all_labels, sent_info = \
                [], [], [], [], []

        for qID, (quest, contexts, labels, gold_idx) in data.iteritems():
            n_contexts = len(contexts)
            quest_info.append((qID, n_contexts, gold_idx))
            all_quests += [quest for _ in range(n_contexts)]
            all_contexts += contexts
            all_labels += labels
            sent_info += [(qID, c_idx) for c_idx in range(n_contexts)]

        return all_quests, all_contexts, all_labels, quest_info, sent_info

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

        assert params.usepytorch, "Must use pytorch (for ranking loss)!"
        config_classifier = {'nclasses': 2, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'cudaEfficient': True,
                             'classifier': params.classifier,
                             'nhid': params.nhid, 'maxepoch': 15,
                             'nepoches': 1, 'noreg': False,
                             'train_rank': True} # was True
        clf = SplitClassifier(self.X, self.y, config_classifier)
        devacc, testacc, devprobs, testprobs = clf.run()

        def topK(quest_info, rankings, k):
            cur_q = 0
            count_k = 0. # number of times gold sentence in top k
            n_sent_k = 0. # number of sentences with at least k sents
            n_trivial = 0. # number of sentences w <= k sents
            for q in quest_info:
                gold_idx, n_sents = q[2], q[1]
                if n_sents > k:
                    n_sent_k += 1.
                    for i in xrange(k):
                        count_k += int(rankings[cur_q+i][2] == gold_idx)
                else:
                    n_trivial += 1
                cur_q += n_sents
            return round(100*count_k / n_sent_k, 2), \
                    round(100*(count_k+n_trivial) / (n_sent_k+n_trivial), 2)

        # group by question, then probability
        val_rankings = [(qID, prob[1], sentID) for (qID, sentID), prob in \
                zip(self.val_sent_info, devprobs)]
        val_rankings.sort(key=lambda z: (z[0], -z[1]))
        te_rankings = [(qID, prob[1], sentID) for (qID, sentID), prob in \
                zip(self.te_sent_info, testprobs)]
        te_rankings.sort(key=lambda z: (z[0], -z[1]))

        # check that I'm not an idiot
        cur_q = 0
        for q in self.val_quest_info:
            q_id, n_sents = q[0], q[1]
            for i in xrange(n_sents):
                assert val_rankings[cur_q+i][0] == q_id
            cur_q += n_sents
        cur_q = 0
        for q in self.te_quest_info:
            q_id, n_sents = q[0], q[1]
            for i in xrange(n_sents):
                assert te_rankings[cur_q+i][0] == q_id
            cur_q += n_sents

        logging.info('Dev acc : {0} Test acc : {1} for SQuAD\n'
                      .format(devacc, testacc))
        logging.info('Dev recall@1: {0} recall@2: {1} recall@3: {2} for SQuAD\n'
                      .format(topK(self.val_quest_info, val_rankings, 1), \
                            topK(self.val_quest_info, val_rankings, 2), \
                            topK(self.val_quest_info, val_rankings, 3)))
        logging.info('Test recall@1: {0} recall@2: {1} recall@3: {2} for SQuAD\n'
                      .format(topK(self.te_quest_info, te_rankings, 1), \
                            topK(self.te_quest_info, te_rankings, 2), \
                            topK(self.te_quest_info, te_rankings, 3)))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
