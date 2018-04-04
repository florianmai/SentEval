''' Some helper functions '''
import nltk
import codecs

def process_sentence(sent, max_seq_len):
    ''' Method to process the string representation of a sentence '''
    return nltk.word_tokenize(sent)[:max_seq_len]

def load_tsv(data_file, max_seq_len, s1_idx=0, s2_idx=1, targ_idx=2, targ_map=None, targ_fn=None,
             skip_rows=0, delimiter='\t'):
    '''Load a tsv'''
    sents1, sents2, targs = [], [], []
    with codecs.open(data_file, 'r', 'utf-8') as data_fh:
        for _ in range(skip_rows):
            data_fh.readline()
        for row_idx, row in enumerate(data_fh):
            try:
                row = row.split(delimiter)
                sent1 = process_sentence(row[s1_idx], max_seq_len)
                if not row[targ_idx] or not sent1:
                    continue
                if targ_map is not None:
                    targ = targ_map[row[targ_idx]]
                elif targ_fn is not None:
                    targ = targ_fn(row[targ_idx])
                else:
                    targ = int(row[targ_idx])
                if s2_idx is not None:
                    sent2 = process_sentence(row[s2_idx], max_seq_len)
                    if not sent2:
                        continue
                    sents2.append(sent2)
                sents1.append(sent1)
                targs.append(targ)
            except Exception as e:
                print(e, row_idx)
                continue
        if s2_idx is not None:
            assert len(sents1) == len(sents2) == len(targs)
            return sents1, sents2, targs
        else:
            assert len(sents1) == len(targs)
            return sents1, targs

def sort_split(split):
    '''Sort a split in decreasing order of sentence length'''
    assert len(split) == 2 or len(split) == 3, 'Invalid number of cols for split!'
    if len(split) == 3:
        sort_key = lambda x: (len(x[0]), len(x[1]), x[2])
    else:
        sort_key = lambda x: (len(x[0]),  x[1])
    return map(list, zip(*sorted(zip(*split), key=sort_key)))

def get_tasks(inp_str):
    '''Get the tasks given a comma separated list'''
    tasks = inp_str.split(',')
    return tasks
