import collections
from collections import Counter
import json

from fastNLP import DataSet, Instance
from fastNLP import Vocabulary
from fastNLP.io import Loader, DataBundle
import numpy as np
import sparse
import unidecode

from data.pipe import UnifyPipe


class RePipe(UnifyPipe):
    def __init__(self, model_name):
        super(RePipe, self).__init__(model_name=model_name)
        self.matrix_segs = {}

    def process(self, data_bundle: DataBundle) -> DataBundle:
        word2bpes = {}
        ent_vocab = Vocabulary(padding=None, unknown=None)
        rel_vocab = Vocabulary(padding=None, unknown=None)

        for ins in data_bundle.get_dataset('train'):
            for _ner in ins['ner']:
                ent_vocab.add_word_lst([_t[-1] for _t in _ner])
            for _rel in ins['relations']:
                rel_vocab.add_word_lst([_t[-1] for _t in _rel])
        distances = Counter()
        valid_jump = {}
        for name in data_bundle.get_dataset_names():
            valid_jump[name] = collections.defaultdict(set)
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            counter = set()
            for ins_idx, ins in enumerate(ds):
                sents, ners, rels = ins['sentences'], ins['ner'], ins['relations']
                cum_len = 0
                for sent, ner, rel in zip(sents, ners, rels):
                    bpes = [self.cls]
                    indexes = [0]
                    for idx, word in enumerate(sent, start=1):
                        if word in word2bpes:
                            _bpes = word2bpes[word]
                        else:
                            _bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                          add_special_tokens=False)
                            word2bpes[word] = _bpes
                        indexes.extend([idx]*len(_bpes))
                        bpes.extend(_bpes)
                    bpes.append(self.sep)
                    indexes.append(0)

                    matrix = np.zeros((len(sent), len(sent), len(ent_vocab)+len(rel_vocab)), dtype=np.int8)
                    ner_dict = {}
                    ent_target = []

                    for _ner in ner:
                        s, e, t = _ner
                        s, e = s-cum_len, e-cum_len
                        # assert ent_matrix[e, s] == 0  # ACE2005不满足这个，同一个ent可能有多个type
                        if matrix[s, e].sum() != 0:
                            counter.add(ins_idx)
                        matrix[s, e, ent_vocab.to_index(t)] = 1
                        matrix[e, s, ent_vocab.to_index(t)] = 1
                        ner_dict[(s, e)] = ent_vocab.to_index(t)
                        ent_target.append((s, e, ent_vocab.to_index(t)))

                    rel_mask = np.zeros((len(sent), len(sent)), dtype=bool)
                    for i in range(len(ent_target)):
                        for j in range(len(ent_target)):
                            if i != j:
                                s1, e1, t1 = ent_target[i]
                                s2, e2, t2 = ent_target[j]
                                rel_mask[s1, s2] = 1
                                rel_mask[e1, e2] = 1

                    rel_target = []
                    shift = len(ent_vocab)
                    for _rel in set([tuple(_rel) for _rel in rel]):
                        s1, e1, s2, e2, r = _rel
                        s1, e1, s2, e2 = s1-cum_len, e1-cum_len, s2-cum_len, e2-cum_len
                        assert (s1, e1) in ner_dict and (s2, e2) in ner_dict
                        assert matrix[s1, e1].any() == 1 and matrix[s2, e2].any() == 1
                        r_idx = rel_vocab.to_index(r)
                        valid_jump[name][(ner_dict[(s1, e1)], ner_dict[(s2, e2)])].add(r_idx)
                        matrix[s1, s2, r_idx+shift] = 1  # 表示从这个ent跳到另一个ent
                        matrix[e1, e2, r_idx+shift] = 1  # 表示从这个ent跳到另一个ent
                        distances[abs(s2-e1)] += 1
                        distances[abs(s1-e2)] += 1
                        rel_target.append((s1, e1, ner_dict[(s1, e1)], s2, e2, ner_dict[(s2, e2)], r_idx))
                    cum_len += len(sent)
                    matrix = sparse.COO.from_numpy(matrix)
                    rel_mask = sparse.COO.from_numpy(rel_mask)
                    new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                                       word_len=len(sent), matrix=matrix, ent_target=ent_target,
                                       rel_target=rel_target, rel_mask=rel_mask)
                    new_ds.append(new_ins)
            print(f"for name:{name}, overlap entity:{counter}")
            data_bundle.set_dataset(new_ds, name=name)
        setattr(data_bundle, 'ent_vocab', ent_vocab)
        print(distances)
        print(valid_jump)
        setattr(data_bundle, 'rel_vocab', rel_vocab)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('ent_target', None)
        data_bundle.set_pad('rel_target', None)
        self.matrix_segs['ent'] = len(ent_vocab)
        self.matrix_segs['rel'] = len(rel_vocab)
        return data_bundle

    def process_from_file(self, paths: str) -> DataBundle:
        symmetric_rels = None
        if 'scierc' in paths.lower():
            symmetric_rels = ['COMPARE', 'CONJUNCTION']
            loader = SciRELoader()
        elif 'ace2005' in paths.lower():
            loader = SciRELoader()
            symmetric_rels = ['PER-SOC']

        dl = loader.load(paths)
        setattr(dl, 'symmetric_rels', symmetric_rels)
        return self.process(dl)


class SciRELoader(Loader):
    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                data.pop('clusters')
                data.pop('doc_key')
                ins = Instance(**data)
                ds.append(ins)
        return ds


class NoEntTypeRePipe(UnifyPipe):
    def __init__(self, model_name):
        super(NoEntTypeRePipe, self).__init__(model_name=model_name)
        self.matrix_segs = {}

    def process(self, data_bundle):
        word2bpes = {}
        ent_vocab = Vocabulary(padding=None, unknown=None)
        rel_vocab = Vocabulary(padding=None, unknown=None)

        for name, ds in data_bundle.iter_datasets():
            for ins in ds:
                ent_vocab.add_word_lst([_t[-1] for _t in ins['raw_ents']])
                rel_vocab.add_word_lst([_t[-1] for _t in ins['raw_rels']])

        for name in data_bundle.get_dataset_names():
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            counter = set()
            for ins_idx, ins in enumerate(ds):
                sent, ner, rel = ins['raw_words'], ins['raw_ents'], ins['raw_rels']
                bpes = [self.cls]
                indexes = [0]
                for idx, word in enumerate(sent, start=1):
                    if word in word2bpes:
                        _bpes = word2bpes[word]
                    else:
                        _bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                      add_special_tokens=False)
                        word2bpes[word] = _bpes
                    indexes.extend([idx]*len(_bpes))
                    bpes.extend(_bpes)
                bpes.append(self.sep)
                indexes.append(0)

                matrix = np.zeros((len(sent), len(sent), len(ent_vocab)+len(rel_vocab)), dtype=np.int8)
                ner_dict = set()
                ent_target = []
                for _ner in ner:
                    s, e, t = _ner
                    # assert ent_matrix[e, s] == 0  # ACE2005不满足这个，同一个ent可能有多个type
                    # if matrix[s, e].sum() != 0:
                    #     counter.add(ins_idx)
                    matrix[s, e, ent_vocab.to_index(t)] = 1  # shift for not entity/next word
                    matrix[e, s, ent_vocab.to_index(t)] = 1  # shift for not entity/next word
                    ner_dict.add((s, e))
                    ent_target.append((s, e, ent_vocab.to_index(t)))

                rel_mask = np.zeros((len(sent), len(sent)), dtype=bool)
                for i in range(len(ent_target)):
                    for j in range(len(ent_target)):
                        if i != j:
                            s1, e1, t1 = ent_target[i]
                            s2, e2, t2 = ent_target[j]
                            rel_mask[s1, s2] = 1
                            rel_mask[e1, e2] = 1

                rel_target = set()
                shift = len(ent_vocab)
                for _rel in set([tuple(_rel) for _rel in rel]):
                    s1, e1, s2, e2, r = _rel
                    assert (s1, e1) in ner_dict and (s2, e2) in ner_dict
                    assert matrix[s1, e1].any() == 1 and matrix[s2, e2].any() == 1
                    r_idx = rel_vocab.to_index(r)
                    matrix[s1, s2, r_idx+shift] = 1  # 表示从这个ent跳到另一个ent
                    matrix[e1, e2, r_idx+shift] = 1  # 表示从这个ent跳到另一个ent
                    # rel_target.append((s1, e1, 0, s2, e2, 0, r_idx))
                    rel_target.add((' '.join(ins['raw_words'][s1:e1+1]), r_idx, ' '.join(ins['raw_words'][s2:e2+1])))
                matrix = sparse.COO.from_numpy(matrix)
                rel_mask = sparse.COO.from_numpy(rel_mask)
                new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                                   word_len=len(sent), matrix=matrix, ent_target=ent_target,
                                   rel_target=rel_target, rel_mask=rel_mask, raw_words=ins['raw_words'])
                new_ds.append(new_ins)
            # print(f"for name:{name}, overlap entity:{counter}({len(counter)}")
            data_bundle.set_dataset(new_ds, name=name)
        setattr(data_bundle, 'ent_vocab', ent_vocab)
        setattr(data_bundle, 'rel_vocab', rel_vocab)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('ent_target', None)
        data_bundle.set_pad('rel_target', None)
        data_bundle.set_pad('raw_words', None)
        self.matrix_segs['ent'] = len(ent_vocab)
        self.matrix_segs['rel'] = len(rel_vocab)
        return data_bundle

    def process_from_file(self, paths: str) -> DataBundle:
        if 'tplinker' in paths:
            loader = WebNLGLoader()
        dl = loader.load(paths)
        return self.process(dl)


class OneRelPipe(UnifyPipe):
    def __init__(self, model_name):
        super(OneRelPipe, self).__init__(model_name=model_name)
        self.matrix_segs = {}

    def process_from_file(self, paths: str, replicate=False) -> DataBundle:
        if replicate:
            dl = ReplicateOneRelLoader().load(paths)
        else:
            dl = OneRelLoader().load(paths)
        return self.process(dl)

    def process(self, data_bundle: DataBundle) -> DataBundle:
        word2bpes = {}
        ent_vocab = Vocabulary(padding=None, unknown=None)
        rel_vocab = Vocabulary(padding=None, unknown=None)

        for name, ds in data_bundle.iter_datasets():
            for ins in ds:
                ent_vocab.add_word_lst([_t[-1] for _t in ins['raw_ents']])
                rel_vocab.add_word_lst([_t[-1] for _t in ins['raw_rels']])

        for name in data_bundle.get_dataset_names():
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            for ins_idx, ins in enumerate(ds):
                sent, ner, rel = ins['raw_words'], ins['raw_ents'], ins['raw_rels']
                bpes = [self.cls]
                indexes = [0]
                for idx, word in enumerate(sent, start=1):
                    if word in word2bpes:
                        _bpes = word2bpes[word]
                    else:
                        _bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                      add_special_tokens=False)
                        word2bpes[word] = _bpes
                    indexes.extend([idx]*len(_bpes))
                    bpes.extend(_bpes)
                bpes.append(self.sep)
                indexes.append(0)

                matrix = np.zeros((len(sent), len(sent), len(ent_vocab)+len(rel_vocab)), dtype=np.int8)
                ner_dict = set()
                ent_target = []
                for _ner in ner:
                    s, e, t = _ner
                    # assert ent_matrix[e, s] == 0  # ACE2005不满足这个，同一个ent可能有多个type
                    # if matrix[s, e].sum() != 0:
                    #     counter.add(ins_idx)
                    matrix[s, e, ent_vocab.to_index(t)] = 1  # shift for not entity/next word
                    matrix[e, s, ent_vocab.to_index(t)] = 1  # shift for not entity/next word
                    ner_dict.add((s, e))
                    ent_target.append((s, e, ent_vocab.to_index(t)))

                rel_mask = np.zeros((len(sent), len(sent)), dtype=bool)
                for i in range(len(ent_target)):
                    for j in range(len(ent_target)):
                        if i != j:
                            s1, e1, t1 = ent_target[i]
                            s2, e2, t2 = ent_target[j]
                            rel_mask[s1, s2] = 1
                            rel_mask[e1, e2] = 1

                rel_target = set()
                shift = len(ent_vocab)
                for _rel in set([tuple(_rel) for _rel in rel]):
                    s1, e1, s2, e2, r = _rel
                    assert (s1, e1) in ner_dict and (s2, e2) in ner_dict
                    assert matrix[s1, e1].any() == 1 and matrix[s2, e2].any() == 1
                    r_idx = rel_vocab.to_index(r)
                    matrix[s1, s2, r_idx+shift] = 1  # 表示从这个ent跳到另一个ent
                    matrix[e1, e2, r_idx+shift] = 1  # 表示从这个ent跳到另一个ent
                    rel_target.add((' '.join(ins['raw_words'][s1:e1+1]), r_idx, ' '.join(ins['raw_words'][s2:e2+1])))
                matrix = sparse.COO.from_numpy(matrix)
                rel_mask = sparse.COO.from_numpy(rel_mask)
                new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                                   word_len=len(sent), matrix=matrix, ent_target=ent_target,
                                   rel_target=list(rel_target), rel_mask=rel_mask, raw_words=ins['raw_words']
                                   )
                new_ds.append(new_ins)
            # print(f"for name:{name}, overlap entity:{counter}({len(counter)}")
            data_bundle.set_dataset(new_ds, name=name)
        setattr(data_bundle, 'ent_vocab', ent_vocab)
        setattr(data_bundle, 'rel_vocab', rel_vocab)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('ent_target', None)
        data_bundle.set_pad('rel_target', None)
        data_bundle.set_pad('raw_words', None)

        self.matrix_segs['ent'] = len(ent_vocab)
        self.matrix_segs['rel'] = len(rel_vocab)
        return data_bundle


class WebNLGLoader(Loader):
    def _load(self, path):
        ds = DataSet()
        with open(path, 'r') as f:
            for data in json.load(f):
                char2wordid = {}
                word_idx = 0
                for idx, c in enumerate(data['text']):
                    if c == ' ':
                        word_idx+=1
                    else:
                        char2wordid[idx] = word_idx
                if c != ' ':
                    char2wordid[idx] = word_idx

                raw_words = data['text'].split()
                rel_lst = data['relation_list']
                ent_lst = data['entity_list']

                raw_ents = set()  # (s, e, t)
                raw_rels = []  # (s1, e1, s2, e2, t)
                ent_dict = set()
                for ent in ent_lst:
                    c_s, c_e = ent['char_span']
                    text = ent['text']
                    s, e = char2wordid[c_s], char2wordid[c_e-1]
                    assert ' '.join(raw_words[s:e+1]) == text, (' '.join(raw_words[s:e+1]), text)
                    ent_dict.add((s, e))

                for rel in rel_lst:
                    s_c_s, s_c_e = rel['subj_char_span']
                    o_c_s, o_c_e = rel['obj_char_span']
                    s_s, s_e = char2wordid[s_c_s], char2wordid[s_c_e-1]
                    c_s, c_e = char2wordid[o_c_s], char2wordid[o_c_e-1]
                    assert ' '.join(raw_words[s_s:s_e+1]) == rel['subject']
                    assert ' '.join(raw_words[c_s:c_e+1]) == rel['object']
                    assert (s_s, s_e) in ent_dict and (c_s, c_e) in ent_dict
                    raw_rels.append((s_s, s_e, c_s, c_e, rel['predicate']))
                    raw_ents.add((s_s, s_e, 'default'))
                    raw_ents.add((c_s, c_e, 'default'))
                ds.append(Instance(raw_words=raw_words, raw_ents=list(raw_ents), raw_rels=raw_rels))
        return ds


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results


class ReplicateOneRelLoader(Loader):
    def _load(self, path):
        ds = DataSet()
        with open(path, 'r') as f:
            data = json.load(f)
            for d in data:
                raw_words = unidecode.unidecode(d['text']).split()
                raw_ents = set()
                raw_rels = set()
                for triple in d['triple_list']:
                    head, predicate, tail = triple
                    # 防止一些音标带来的问题
                    heads, tails = unidecode.unidecode(head).split(), unidecode.unidecode(tail).split()
                    head_idxes = find_sub_list(heads, raw_words)
                    tail_idxes = find_sub_list(tails, raw_words)
                    if len(tail_idxes)==0 or len(head_idxes) == 0:
                        import pdb
                        pdb.set_trace()
                    else:
                        for _head_idxes in head_idxes:
                            for _tail_idxes in tail_idxes:
                                raw_ents.add((_head_idxes[0], _head_idxes[1], 'default'))
                                raw_ents.add((_tail_idxes[0], _tail_idxes[1], 'default'))
                                raw_rels.add((_head_idxes[0], _head_idxes[1], _tail_idxes[0], _tail_idxes[1], predicate))
                # 根据onerel评测时直接取set，所以这里也直接取了set了
                ds.append(Instance(raw_words=raw_words, raw_ents=list(raw_ents), raw_rels=list(raw_rels),
                                   raw_triples=d['triple_list']))
        return ds


class OneRelLoader(Loader):
    def _load(self, path):
        ds = DataSet()
        with open(path, 'r') as f:
            data = json.load(f)
            for d in data:
                raw_words = unidecode.unidecode(d['text']).split()
                raw_ents = set()
                raw_rels = set()
                for triple in d['triple_list']:
                    head, predicate, tail = triple
                    # 防止一些音标带来的问题
                    heads, tails = unidecode.unidecode(head).split(), unidecode.unidecode(tail).split()
                    head_idxes = find_sub_list(heads, raw_words)
                    tail_idxes = find_sub_list(tails, raw_words)
                    if len(tail_idxes)==0 or len(head_idxes) == 0:
                        import pdb
                        pdb.set_trace()
                    else:
                        raw_ents.add((head_idxes[0][0], head_idxes[0][1], 'default'))
                        raw_ents.add((tail_idxes[0][0], tail_idxes[0][1], 'default'))
                        raw_rels.add((head_idxes[0][0], head_idxes[0][1], tail_idxes[0][0], tail_idxes[0][1], predicate))
                # 根据onerel评测时直接取set，所以这里也直接取了set了
                ds.append(Instance(raw_words=raw_words, raw_ents=list(raw_ents), raw_rels=list(raw_rels),
                                   raw_triples=d['triple_list']))
        return ds


if __name__ == '__main__':
    # webnlg_pipe = RePipe('roberta-base')
    # dl = webnlg_pipe.process_from_file('/remote-home/hyan01/exps/RelationExtraction/data/tplinker_webnlg')
    paths = '/remote-home/hyan01/exps/RelationExtraction/data/UniRE_SciERC'
    pipe = RePipe('roberta-base')
    # paths = '/remote-home/hyan01/exps/RelationExtraction/data/UniRE_ace2005'
    pipe.process_from_file(paths)
