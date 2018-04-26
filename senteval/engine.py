# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals
import ipdb as pdb

from senteval import utils
from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
from senteval.snli import SNLIEval
from senteval.mnli import MNLIEval
from senteval.trec import TRECEval
from senteval.sick import SICKRelatednessEval, SICKEntailmentEval
from senteval.mrpc import MRPCEval
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval
from senteval.sst import SSTEval
from senteval.rank import ImageCaptionRetrievalEval
from senteval.quora import QuoraEval
from senteval.rte import RTEEval
from senteval.squad import SQuADEval
from senteval.warstadt import WarstadtEval
from senteval.wnli import WNLIEval
from senteval.anli import ANLIEval

ALL_TASKS = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
             'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
             'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
             'STS14', 'STS15', 'STS16',
             'MNLI', 'Quora', 'RTE', 'SQuAD', 'Warstadt', 'WNLI', 'ANLI']
BENCHMARK_TASKS = ['SST2', 'MRPC', 'STSBenchmark', 'MNLI', 'Quora', 'RTE', 'SQuAD', 'Warstadt', 'WNLI']

class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.max_seq_len = 50 if 'max_seq_len' not in params else params.max_seq_len
        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ALL_TASKS
        self.evaluation, self.results = None, None

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if isinstance(name, list):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)
        max_seq_len = self.params.max_seq_len
        load_data = self.params.load_data
        seed = self.params.seed
        if name == 'CR':
            self.evaluation = CREval(tpath + '/CR', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'MR':
            self.evaluation = MREval(tpath + '/MR', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'MPQA':
            self.evaluation = MPQAEval(tpath + '/MPQA', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'SUBJ':
            self.evaluation = SUBJEval(tpath + '/SUBJ', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'SST2':
            self.evaluation = SSTEval(tpath + '/SST/binary', nclasses=2, max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'SST5':
            self.evaluation = SSTEval(tpath + '/SST/fine', nclasses=5, max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'TREC':
            self.evaluation = TRECEval(tpath + '/TREC', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'MRPC':
            self.evaluation = MRPCEval(tpath + '/MRPC', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'SICKRelatedness':
            self.evaluation = SICKRelatednessEval(tpath + '/SICK', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'STSBenchmark':
            self.evaluation = STSBenchmarkEval(tpath + '/STS/STSBenchmark', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'SICKEntailment':
            self.evaluation = SICKEntailmentEval(tpath + '/SICK', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'SNLI':
            self.evaluation = SNLIEval(tpath + '/SNLI', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            fpath = name + '-en-test'
            self.evaluation = eval(name + 'Eval')(tpath + '/STS/' + fpath, max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'ImageCaptionRetrieval':
            self.evaluation = ImageCaptionRetrievalEval(tpath + '/COCO', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'MNLI':
            self.evaluation = MNLIEval(tpath + '/MNLI', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'Quora':
            self.evaluation = QuoraEval(tpath + '/Quora', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'RTE':
            self.evaluation = RTEEval(tpath + '/RTE', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'SQuAD':
            self.evaluation = SQuADEval(tpath + '/SQuAD', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'WNLI':
            self.evaluation = WNLIEval(tpath + '/WNLI', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'Warstadt':
            #self.evaluation = WarstadtEval(tpath + '/Warstadt_old', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
            self.evaluation = WarstadtEval(tpath + '/Warstadt', max_seq_len=max_seq_len, load_data=load_data, seed=seed)
        elif name == 'ANLI':
            self.evaluation = ANLIEval(tpath + '/ANLI', max_seq_len=max_seq_len, load_data=load_data, seed=seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
