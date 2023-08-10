import collections
from collections import Counter
import json

from fastNLP import DataSet, Instance
from fastNLP import Vocabulary
from fastNLP.io import Loader, DataBundle
import numpy as np
import sparse

from data.pipe import UnifyPipe


class _RePipe(UnifyPipe):
    # 左右平等地加入信息
    def __init__(self, model_name, use_sym=False, max_length=256):
        super(_RePipe, self).__init__(model_name=model_name)
        self.matrix_segs = {}
        self.use_sym = use_sym
        self.max_length = max_length

    def process(self, data_bundle: DataBundle) -> DataBundle:
        word2bpes = {}
        ent_vocab = Vocabulary(padding=None, unknown=None)
        rel_vocab = Vocabulary(padding=None, unknown=None)
        sys_rels = set()
        for ins in data_bundle.get_dataset('train'):
            for _ner in ins['ner']:
                ent_vocab.add_word_lst([_t[-1] for _t in _ner])
            for _rel in ins['relations']:
                rel_vocab.add_word_lst([_t[-1] for _t in _rel])
                if self.use_sym:
                    for _t in _rel:
                        if _t[-1] in data_bundle.symmetric_rels:
                            sys_rels.add(rel_vocab.to_index(_t[-1]))
        if self.use_sym:
            print(f"Found {len(sys_rels)} symetric relations.")
            setattr(data_bundle, 'sys_rels', sys_rels)
        distances = Counter()
        valid_jump = {}
        for name in data_bundle.get_dataset_names():
            valid_jump[name] = collections.defaultdict(set)
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            counter = set()
            for ins_idx, ins in enumerate(ds):
                sents, ners, rels = ins['sentences'], ins['ner'], ins['relations']
                if self.max_length != 0:
                    sents_star_ends = []
                    sent_bpes = []
                    for sent in sents:
                        sents_star_ends.append([len(sent_bpes)])
                        for idx, word in enumerate(sent, start=1):
                            if word in word2bpes:
                                _bpes = word2bpes[word]
                            else:
                                _bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                              add_special_tokens=False)
                                word2bpes[word] = _bpes
                            sent_bpes.extend(_bpes)
                        sents_star_ends[-1].append(len(sent_bpes))
                cum_len = 0

                for sent_idx, (sent, ner, rel) in enumerate(zip(sents, ners, rels)):
                    bpes = [self.cls]
                    indexes = [0]
                    for idx, word in enumerate(sent, start=1):
                        if word in word2bpes:
                            _bpes = word2bpes[word]
                        else:
                            _bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                          add_special_tokens=False)
                            word2bpes[word] = _bpes
                        indexes.extend([idx] * len(_bpes))
                        bpes.extend(_bpes)
                    bpes.append(self.sep)
                    indexes.append(0)
                    if self.max_length != 0:
                        sent_start, sent_end = sents_star_ends[sent_idx]
                        context_size = (self.max_length - len(bpes)) // 2
                        if context_size < 0:
                            raise RuntimeError()
                        left_start = max(sent_start - context_size, 0)
                        right_end = min(len(sent_bpes), sent_end + context_size)
                        left_bpes = sent_bpes[left_start:sent_start]
                        right_bpes = sent_bpes[sent_end:right_end]
                        indexes = [0] * len(left_bpes) + indexes + [0] * len(right_bpes)
                        bpes = bpes[:1] + left_bpes + bpes[1:-1] + right_bpes + bpes[-1:]

                    matrix = np.zeros((len(sent), len(sent), len(ent_vocab) + len(rel_vocab)), dtype=np.int8)
                    ner_dict = {}
                    ent_target = []

                    for _ner in ner:
                        s, e, t = _ner
                        s, e = s - cum_len, e - cum_len
                        # assert ent_matrix[e, s] == 0  # ACE2005不满足这个，同一个ent可能有多个type
                        if matrix[s, e].sum() != 0:
                            counter.add(ins_idx)
                        matrix[s, e, ent_vocab.to_index(t)] = 1  # shift for not entity/next word
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
                                rel_mask[s2, s1] = 1
                                rel_mask[e2, e1] = 1

                    rel_target = []
                    shift = len(ent_vocab)
                    for _rel in set([tuple(_rel) for _rel in rel]):
                        s1, e1, s2, e2, r = _rel
                        s1, e1, s2, e2 = s1 - cum_len, e1 - cum_len, s2 - cum_len, e2 - cum_len
                        assert (s1, e1) in ner_dict and (s2, e2) in ner_dict
                        assert matrix[s1, e1].any() == 1 and matrix[s2, e2].any() == 1
                        r_idx = rel_vocab.to_index(r)
                        valid_jump[name][(ner_dict[(s1, e1)], ner_dict[(s2, e2)])].add(r_idx)
                        matrix[s1, s2, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                        matrix[e1, e2, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                        if r_idx in sys_rels:
                            matrix[s2, s1, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                            matrix[e2, e1, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                            rel_target.append((s2, e2, ner_dict[(s2, e2)], s1, e1, ner_dict[(s1, e1)], r_idx))
                        distances[abs(s2 - e1)] += 1
                        distances[abs(s1 - e2)] += 1
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
            symmetric_rels = {'COMPARE', 'CONJUNCTION'}
            loader = SciRELoader()
        elif 'ace2005' in paths.lower():
            loader = SciRELoader()
            symmetric_rels = {'PER-SOC'}

        dl = loader.load(paths)
        setattr(dl, 'symmetric_rels', symmetric_rels)
        return self.process(dl)


class RePipe_(UnifyPipe):
    # 加入context的时候一定尽量加入到256，如果左边少一点，就让右边多一点
    def __init__(self, model_name, use_sym=False, max_length=256):
        super(RePipe_, self).__init__(model_name=model_name)
        self.matrix_segs = {}
        self.use_sym = use_sym
        self.max_length = max_length

    def process(self, data_bundle: DataBundle) -> DataBundle:
        word2bpes = {}
        ent_vocab = Vocabulary(padding=None, unknown=None)
        rel_vocab = Vocabulary(padding=None, unknown=None)
        sys_rels = set()
        for ins in data_bundle.get_dataset('train'):
            for _ner in ins['ner']:
                ent_vocab.add_word_lst([_t[-1] for _t in _ner])
            for _rel in ins['relations']:
                rel_vocab.add_word_lst([_t[-1] for _t in _rel])
                if self.use_sym:
                    for _t in _rel:
                        if _t[-1] in data_bundle.symmetric_rels:
                            sys_rels.add(rel_vocab.to_index(_t[-1]))
        if self.use_sym:
            print(f"Found {len(sys_rels)} symetric relations.")
            setattr(data_bundle, 'sys_rels', sys_rels)
        distances = Counter()
        valid_jump = {}
        for name in data_bundle.get_dataset_names():
            valid_jump[name] = collections.defaultdict(set)
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            counter = set()
            for ins_idx, ins in enumerate(ds):
                sents, ners, rels = ins['sentences'], ins['ner'], ins['relations']
                if self.max_length != 0:
                    sents_star_ends = []
                    sent_bpes = []
                    for sent in sents:
                        sents_star_ends.append([len(sent_bpes)])
                        for idx, word in enumerate(sent, start=1):
                            if word in word2bpes:
                                _bpes = word2bpes[word]
                            else:
                                _bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                              add_special_tokens=False)
                                word2bpes[word] = _bpes
                            sent_bpes.extend(_bpes)
                        sents_star_ends[-1].append(len(sent_bpes))
                cum_len = 0

                for sent_idx, (sent, ner, rel) in enumerate(zip(sents, ners, rels)):
                    bpes = [self.cls]
                    indexes = [0]
                    for idx, word in enumerate(sent, start=1):
                        if word in word2bpes:
                            _bpes = word2bpes[word]
                        else:
                            _bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                          add_special_tokens=False)
                            word2bpes[word] = _bpes
                        indexes.extend([idx] * len(_bpes))
                        bpes.extend(_bpes)
                    bpes.append(self.sep)
                    indexes.append(0)
                    if self.max_length != 0:
                        sent_start, sent_end = sents_star_ends[sent_idx]
                        sentence_length = sent_end - sent_start
                        context_size = (self.max_length - len(bpes)) // 2
                        if context_size < 0:
                            raise RuntimeError()

                        # follow preprocess from https://github.dev/thunlp/PL-Marker/blob/master/run_re.py
                        left_length = sent_start
                        right_length = len(sent_bpes) - sent_end
                        if left_length < right_length:
                            left_context_length = min(left_length, context_size)
                            right_context_length = min(right_length,
                                                       self.max_length - left_context_length - sentence_length)
                        else:
                            right_context_length = min(right_length, context_size)
                            left_context_length = min(left_length,
                                                      self.max_length - right_context_length - sentence_length)

                        left_start = max(sent_start - left_context_length, 0)
                        right_end = min(len(sent_bpes), sent_end + right_context_length)
                        left_bpes = sent_bpes[left_start:sent_start]
                        right_bpes = sent_bpes[sent_end:right_end]
                        indexes = [0] * len(left_bpes) + indexes + [0] * len(right_bpes)
                        bpes = bpes[:1] + left_bpes + bpes[1:-1] + right_bpes + bpes[-1:]
                    assert len(bpes) <= self.max_length + 2

                    matrix = np.zeros((len(sent), len(sent), len(ent_vocab) + len(rel_vocab)), dtype=np.int8)
                    ner_dict = {}
                    ent_target = []

                    for _ner in ner:
                        s, e, t = _ner
                        s, e = s - cum_len, e - cum_len
                        # assert ent_matrix[e, s] == 0  # ACE2005不满足这个，同一个ent可能有多个type
                        if matrix[s, e].sum() != 0:
                            counter.add(ins_idx)
                        matrix[s, e, ent_vocab.to_index(t)] = 1  # shift for not entity/next word
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
                                rel_mask[s2, s1] = 1
                                rel_mask[e2, e1] = 1

                    rel_target = []
                    shift = len(ent_vocab)
                    for _rel in set([tuple(_rel) for _rel in rel]):
                        s1, e1, s2, e2, r = _rel
                        s1, e1, s2, e2 = s1 - cum_len, e1 - cum_len, s2 - cum_len, e2 - cum_len
                        assert (s1, e1) in ner_dict and (s2, e2) in ner_dict
                        assert matrix[s1, e1].any() == 1 and matrix[s2, e2].any() == 1
                        r_idx = rel_vocab.to_index(r)
                        valid_jump[name][(ner_dict[(s1, e1)], ner_dict[(s2, e2)])].add(r_idx)
                        matrix[s1, s2, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                        matrix[e1, e2, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                        if r_idx in sys_rels:
                            matrix[s2, s1, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                            matrix[e2, e1, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                            rel_target.append((s2, e2, ner_dict[(s2, e2)], s1, e1, ner_dict[(s1, e1)], r_idx))
                        distances[abs(s2 - e1)] += 1
                        distances[abs(s1 - e2)] += 1
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
            symmetric_rels = {'COMPARE', 'CONJUNCTION'}
            loader = SciRELoader()
        elif 'ace2005' in paths.lower():
            loader = SciRELoader()
            symmetric_rels = {'PER-SOC'}

        dl = loader.load(paths)
        setattr(dl, 'symmetric_rels', symmetric_rels)
        return self.process(dl)


class RRePipe_(UnifyPipe):
    # 加入context的时候一定尽量加入到256，如果左边少一点，就让右边多一点. 如果是非对称的关系，就加入一种反关系
    def __init__(self, model_name, use_sym=False, max_length=256):
        super(RRePipe_, self).__init__(model_name=model_name)
        self.matrix_segs = {}
        self.use_sym = use_sym
        self.max_length = max_length

    def process(self, data_bundle: DataBundle) -> DataBundle:
        word2bpes = {}
        ent_vocab = Vocabulary(padding=None, unknown=None)
        rel_vocab = Vocabulary(padding=None, unknown=None)
        for rel in data_bundle.symmetric_rels:
            rel_vocab.add_word(rel)
        _start = len(rel_vocab)  # 保证一定在前面
        sys_rels = set()
        for ins in data_bundle.get_dataset('train'):
            for _ner in ins['ner']:
                ent_vocab.add_word_lst([_t[-1] for _t in _ner])
            for _rel in ins['relations']:
                rel_vocab.add_word_lst([_t[-1] for _t in _rel])
                if self.use_sym:
                    for _t in _rel:
                        if _t[-1] in data_bundle.symmetric_rels:
                            sys_rels.add(rel_vocab.to_index(_t[-1]))
        if self.use_sym:
            print(f"Found {len(sys_rels)} symetric relations.")
            setattr(data_bundle, 'sys_rels', sys_rels)
        self.matrix_segs['r_rel'] = len(rel_vocab) - len(sys_rels)
        for rel in data_bundle.symmetric_rels:
            assert rel_vocab.to_index(rel) < _start
        distances = Counter()
        valid_jump = {}
        for name in data_bundle.get_dataset_names():
            valid_jump[name] = collections.defaultdict(set)
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            counter = set()
            for ins_idx, ins in enumerate(ds):
                sents, ners, rels = ins['sentences'], ins['ner'], ins['relations']
                if self.max_length != 0:
                    sents_star_ends = []
                    sent_bpes = []
                    for sent in sents:
                        sents_star_ends.append([len(sent_bpes)])
                        for idx, word in enumerate(sent, start=1):
                            if word in word2bpes:
                                _bpes = word2bpes[word]
                            else:
                                _bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                              add_special_tokens=False)
                                word2bpes[word] = _bpes
                            sent_bpes.extend(_bpes)
                        sents_star_ends[-1].append(len(sent_bpes))
                cum_len = 0

                for sent_idx, (sent, ner, rel) in enumerate(zip(sents, ners, rels)):
                    bpes = [self.cls]
                    indexes = [0]
                    for idx, word in enumerate(sent, start=1):
                        if word in word2bpes:
                            _bpes = word2bpes[word]
                        else:
                            _bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                          add_special_tokens=False)
                            word2bpes[word] = _bpes
                        indexes.extend([idx] * len(_bpes))
                        bpes.extend(_bpes)
                    bpes.append(self.sep)
                    indexes.append(0)
                    if self.max_length != 0:
                        sent_start, sent_end = sents_star_ends[sent_idx]
                        sentence_length = sent_end - sent_start
                        context_size = (self.max_length - len(bpes)) // 2
                        if context_size < 0:
                            raise RuntimeError()

                        # follow preprocess from https://github.dev/thunlp/PL-Marker/blob/master/run_re.py
                        left_length = sent_start
                        right_length = len(sent_bpes) - sent_end
                        if left_length < right_length:
                            left_context_length = min(left_length, context_size)
                            right_context_length = min(right_length,
                                                       self.max_length - left_context_length - sentence_length)
                        else:
                            right_context_length = min(right_length, context_size)
                            left_context_length = min(left_length,
                                                      self.max_length - right_context_length - sentence_length)

                        left_start = max(sent_start - left_context_length, 0)
                        right_end = min(len(sent_bpes), sent_end + right_context_length)
                        left_bpes = sent_bpes[left_start:sent_start]
                        right_bpes = sent_bpes[sent_end:right_end]
                        indexes = [0] * len(left_bpes) + indexes + [0] * len(right_bpes)
                        bpes = bpes[:1] + left_bpes + bpes[1:-1] + right_bpes + bpes[-1:]
                    assert len(bpes) <= self.max_length + 2

                    matrix = np.zeros((len(sent), len(sent), len(ent_vocab) + len(rel_vocab) +
                                       len(rel_vocab) - len(sys_rels)), dtype=np.int8)
                    ner_dict = {}
                    ent_target = []

                    for _ner in ner:
                        s, e, t = _ner
                        s, e = s - cum_len, e - cum_len
                        # assert ent_matrix[e, s] == 0  # ACE2005不满足这个，同一个ent可能有多个type
                        if matrix[s, e].sum() != 0:
                            counter.add(ins_idx)
                        matrix[s, e, ent_vocab.to_index(t)] = 1  # shift for not entity/next word
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
                                rel_mask[s2, s1] = 1
                                rel_mask[e2, e1] = 1

                    rel_target = []
                    shift = len(ent_vocab)
                    asy_rel_shift = len(ent_vocab) + len(rel_vocab)
                    for _rel in set([tuple(_rel) for _rel in rel]):
                        s1, e1, s2, e2, r = _rel
                        s1, e1, s2, e2 = s1 - cum_len, e1 - cum_len, s2 - cum_len, e2 - cum_len
                        assert (s1, e1) in ner_dict and (s2, e2) in ner_dict
                        assert matrix[s1, e1].any() == 1 and matrix[s2, e2].any() == 1
                        r_idx = rel_vocab.to_index(r)
                        valid_jump[name][(ner_dict[(s1, e1)], ner_dict[(s2, e2)])].add(r_idx)
                        matrix[s1, s2, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                        matrix[e1, e2, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                        if r_idx in sys_rels:
                            matrix[s2, s1, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                            matrix[e2, e1, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                            rel_target.append((s2, e2, ner_dict[(s2, e2)], s1, e1, ner_dict[(s1, e1)], r_idx))
                        else:
                            matrix[s2, s1, asy_rel_shift + r_idx - _start] = 1  # 表示反向的关系
                            matrix[e2, e1, asy_rel_shift + r_idx - _start] = 1  # 表示反向的关系
                        distances[abs(s2 - e1)] += 1
                        distances[abs(s1 - e2)] += 1
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
            symmetric_rels = {'COMPARE', 'CONJUNCTION'}
            loader = SciRELoader()
        elif 'ace2005' in paths.lower():
            loader = SciRELoader()
            symmetric_rels = {'PER-SOC'}

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
                        _bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                      add_special_tokens=False)
                        word2bpes[word] = _bpes
                    indexes.extend([idx] * len(_bpes))
                    bpes.extend(_bpes)
                bpes.append(self.sep)
                indexes.append(0)

                matrix = np.zeros((len(sent), len(sent), len(ent_vocab) + len(rel_vocab)), dtype=np.int8)
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

                rel_target = []
                shift = len(ent_vocab)
                for _rel in set([tuple(_rel) for _rel in rel]):
                    s1, e1, s2, e2, r = _rel
                    assert (s1, e1) in ner_dict and (s2, e2) in ner_dict
                    assert matrix[s1, e1].any() == 1 and matrix[s2, e2].any() == 1
                    r_idx = rel_vocab.to_index(r)
                    matrix[s1, s2, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                    matrix[e1, e2, r_idx + shift] = 1  # 表示从这个ent跳到另一个ent
                    rel_target.append((s1, e1, 0, s2, e2, 0, r_idx))
                matrix = sparse.COO.from_numpy(matrix)
                rel_mask = sparse.COO.from_numpy(rel_mask)
                new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                                   word_len=len(sent), matrix=matrix, ent_target=ent_target,
                                   rel_target=rel_target, rel_mask=rel_mask)
                new_ds.append(new_ins)
            # print(f"for name:{name}, overlap entity:{counter}({len(counter)}")
            data_bundle.set_dataset(new_ds, name=name)
        setattr(data_bundle, 'ent_vocab', ent_vocab)
        setattr(data_bundle, 'rel_vocab', rel_vocab)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('ent_target', None)
        data_bundle.set_pad('rel_target', None)
        self.matrix_segs['ent'] = len(ent_vocab)
        self.matrix_segs['rel'] = len(rel_vocab)
        return data_bundle

    def process_from_file(self, paths: str) -> DataBundle:
        if 'tplinker' in paths:
            loader = WebNLGLoader()
        dl = loader.load(paths)
        return self.process(dl)


class WebNLGLoader(Loader):
    def _load(self, path):
        ds = DataSet()
        with open(path, 'r') as f:
            for data in json.load(f):
                char2wordid = {}
                word_idx = 0
                for idx, c in enumerate(data['text']):
                    if c == ' ':
                        word_idx += 1
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
                    s, e = char2wordid[c_s], char2wordid[c_e - 1]
                    assert ' '.join(raw_words[s:e + 1]) == text, (' '.join(raw_words[s:e + 1]), text)
                    ent_dict.add((s, e))

                for rel in rel_lst:
                    s_c_s, s_c_e = rel['subj_char_span']
                    o_c_s, o_c_e = rel['obj_char_span']
                    s_s, s_e = char2wordid[s_c_s], char2wordid[s_c_e - 1]
                    c_s, c_e = char2wordid[o_c_s], char2wordid[o_c_e - 1]
                    assert ' '.join(raw_words[s_s:s_e + 1]) == rel['subject']
                    assert ' '.join(raw_words[c_s:c_e + 1]) == rel['object']
                    assert (s_s, s_e) in ent_dict and (c_s, c_e) in ent_dict
                    raw_rels.append((s_s, s_e, c_s, c_e, rel['predicate']))
                    raw_ents.add((s_s, s_e, 'default'))
                    raw_ents.add((c_s, c_e, 'default'))
                ds.append(Instance(raw_words=raw_words, raw_ents=list(raw_ents), raw_rels=raw_rels))
        return ds


if __name__ == '__main__':
    # webnlg_pipe = RePipe('roberta-base')
    # dl = webnlg_pipe.process_from_file('/remote-home/hyan01/exps/RelationExtraction/data/tplinker_webnlg')
    paths = '/remote-home/hyan01/exps/RelationExtraction/data/UniRE_SciERC'
    pipe = _RePipe('roberta-base')
    # paths = '/remote-home/hyan01/exps/RelationExtraction/data/UniRE_ace2005'
    pipe.process_from_file(paths)
