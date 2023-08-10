import collections
from collections import defaultdict
import json

from fastNLP import DataSet, Instance
from fastNLP import Vocabulary
from fastNLP.io import Loader, DataBundle
import numpy as np
import sparse

from data.pipe import UnifyPipe


class EEPipe(UnifyPipe):
    def __init__(self, model_name):
        super(EEPipe, self).__init__(model_name=model_name)
        self.matrix_segs = {}

    def process(self, data_bundle: DataBundle) -> DataBundle:
        word2bpes = {}
        tri_vocab = Vocabulary(padding=None, unknown=None)
        role_vocab = Vocabulary(padding=None, unknown=None)

        for ins in data_bundle.get_dataset('train'):
            for t in ins['raw_tris']:
                tri_vocab.add_word(t[-1])
            for args in ins['raw_args']:
                role_vocab.add_word_lst([_t[-1] for _t in args])
        # constrain: {key: set()}  其中key为event_type的index，value是这个event_type运行的role有哪些
        constrain = defaultdict(set)  #
        for name in data_bundle.get_dataset_names():
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            for ins_idx, ins in enumerate(ds):
                sent, raw_tris, raw_args = ins['raw_words'], ins['raw_tris'], ins['raw_args']
                bpes = [self.cls]
                indexes = [0]
                for idx, word in enumerate(sent, start=1):
                    if word in word2bpes:
                        _bpes = word2bpes[word]
                    else:
                        _bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                      add_special_tokens=False)[:5]
                        word2bpes[word] = _bpes
                    indexes.extend([idx] * len(_bpes))
                    bpes.extend(_bpes)
                bpes.append(self.sep)
                if len(bpes) > 300 and name == 'train':
                    print(f"Skip for {name} because {len(bpes)} length")
                    continue
                assert len(bpes) <= 512
                indexes.append(0)

                matrix = np.zeros((len(sent), len(sent), len(tri_vocab) + len(role_vocab) + 1), dtype=np.int8)
                tri_target = []
                for tri in raw_tris:
                    s, e, t = tri
                    assert matrix[s, e, tri_vocab.to_index(t) + 1] == 0  # trigger 不会overlap
                    matrix[s, e, tri_vocab.to_index(t) + 1] = 1
                    matrix[e, s, tri_vocab.to_index(t) + 1] = 1
                    tri_target.append((s, e, tri_vocab.to_index(t)))

                arg_target = []
                roles = []
                checker = collections.Counter()
                for tri, args in zip(raw_tris, raw_args):
                    s, e, t = tri
                    t = tri_vocab.to_index(t)
                    for arg in args:
                        a_s, a_e, a = arg
                        a = role_vocab.to_index(a) + len(tri_vocab) + 1
                        matrix[a_s, a_e, 0] = 1
                        matrix[a_e, a_s, 0] = 1
                        matrix[s, a_s, a] = 1
                        matrix[e, a_e, a] = 1
                        matrix[a_s, s, a] = 1
                        matrix[a_e, e, a] = 1
                        constrain[t].add(a - 1 - len(tri_vocab))
                        roles.append((a_s, a_e))
                        arg_target.append((a_s, a_e, a - 1 - len(tri_vocab), t))
                        checker[arg_target[-1]] += 1

                # 用来给没有关系地方进行采样处理的
                rel_mask = np.zeros((len(sent), len(sent)), dtype=bool)
                tmp = arg_target + tri_target
                for i in range(len(tmp)):
                    for j in range(len(tmp)):
                        if i == j:
                            continue
                        s1, e1 = tmp[i][0], tmp[i][1]
                        s2, e2 = tmp[j][0], tmp[j][1]
                        rel_mask[s1, s2] = 1
                        rel_mask[e1, e2] = 1
                        rel_mask[s2, s1] = 1
                        rel_mask[e2, e1] = 1

                matrix = sparse.COO.from_numpy(matrix)
                rel_mask = sparse.COO.from_numpy(rel_mask)
                new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                                   word_len=len(sent), matrix=matrix, tri_target=tri_target,
                                   arg_target=arg_target, rel_mask=rel_mask, sent_id=ins['sent_id'])
                new_ds.append(new_ins)
            data_bundle.set_dataset(new_ds, name=name)
        setattr(data_bundle, 'tri_vocab', tri_vocab)
        setattr(data_bundle, 'role_vocab', role_vocab)
        setattr(data_bundle, 'constrain', constrain)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('tri_target', None)
        data_bundle.set_pad('arg_target', None)
        self.matrix_segs['arg'] = 1
        self.matrix_segs['role'] = len(role_vocab)
        self.matrix_segs['tri'] = len(tri_vocab)
        return data_bundle

    def process_from_file(self, paths: str, o=False) -> DataBundle:
        if o:
            loader = OACE2005Loader()
        else:
            loader = ACE2005Loader()
        dl = loader.load(paths)
        return self.process(dl)


class EEPipe_(UnifyPipe):
    def __init__(self, model_name):
        super(EEPipe_, self).__init__(model_name=model_name)
        self.matrix_segs = {}

    def process(self, data_bundle: DataBundle) -> DataBundle:
        word2bpes = {}
        tri_vocab = Vocabulary(padding=None, unknown=None)
        role_vocab = Vocabulary(padding=None, unknown=None)

        for ins in data_bundle.get_dataset('train'):
            for t in ins['raw_tris']:
                tri_vocab.add_word(t[-1])
            for args in ins['raw_args']:
                role_vocab.add_word_lst([_t[-1] for _t in args])
        # constrain: {key: set()}  其中key为event_type的index，value是这个event_type运行的role有哪些
        constrain = defaultdict(set)  #
        for name in data_bundle.get_dataset_names():
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            for ins_idx, ins in enumerate(ds):
                sent, raw_tris, raw_args = ins['raw_words'], ins['raw_tris'], ins['raw_args']
                bpes = [self.cls]
                indexes = [0]
                for idx, word in enumerate(sent, start=1):
                    if word in word2bpes:
                        _bpes = word2bpes[word]
                    else:
                        _bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                      add_special_tokens=False)[:5]
                        word2bpes[word] = _bpes
                    indexes.extend([idx] * len(_bpes))
                    bpes.extend(_bpes)
                bpes.append(self.sep)
                if len(bpes) > 300 and name == 'train':
                    print(f"Skip for {name} because {len(bpes)} length")
                    continue
                assert len(bpes) <= 512
                indexes.append(0)

                matrix = np.zeros((len(sent), len(sent), len(tri_vocab) + len(role_vocab) + 1), dtype=np.int8)
                tri_target = []
                for tri in raw_tris:
                    s, e, t = tri
                    assert matrix[s, e, tri_vocab.to_index(t) + 1] == 0  # trigger 不会overlap
                    matrix[s, e, tri_vocab.to_index(t) + 1] = 1
                    matrix[e, s, tri_vocab.to_index(t) + 1] = 1
                    tri_target.append((s, e, tri_vocab.to_index(t)))

                arg_target = []
                roles = []
                checker = collections.Counter()
                for tri, args in zip(raw_tris, raw_args):
                    s, e, t = tri
                    t = tri_vocab.to_index(t)
                    for arg in args:
                        a_s, a_e, a = arg
                        a = role_vocab.to_index(a) + len(tri_vocab) + 1
                        matrix[a_s, a_e, 0] = 1
                        matrix[a_e, a_s, 0] = 1
                        matrix[s, a_s, a] = 1
                        matrix[e, a_e, a] = 1
                        constrain[t].add(a - 1 - len(tri_vocab))
                        roles.append((a_s, a_e))
                        arg_target.append((a_s, a_e, a - 1 - len(tri_vocab), t))
                        checker[arg_target[-1]] += 1

                # 用来给没有关系地方进行采样处理的
                rel_mask = np.zeros((len(sent), len(sent)), dtype=bool)
                tmp = arg_target + tri_target
                for i in range(len(tmp)):
                    for j in range(len(tmp)):
                        if i == j:
                            continue
                        s1, e1 = tmp[i][0], tmp[i][1]
                        s2, e2 = tmp[j][0], tmp[j][1]
                        rel_mask[s1, s2] = 1
                        rel_mask[e1, e2] = 1

                matrix = sparse.COO.from_numpy(matrix)
                rel_mask = sparse.COO.from_numpy(rel_mask)
                new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                                   word_len=len(sent), matrix=matrix, tri_target=tri_target,
                                   arg_target=arg_target, rel_mask=rel_mask, sent_id=ins['sent_id'])
                new_ds.append(new_ins)
            data_bundle.set_dataset(new_ds, name=name)
        setattr(data_bundle, 'tri_vocab', tri_vocab)
        setattr(data_bundle, 'role_vocab', role_vocab)
        setattr(data_bundle, 'constrain', constrain)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('tri_target', None)
        data_bundle.set_pad('arg_target', None)
        self.matrix_segs['arg'] = 1
        self.matrix_segs['role'] = len(role_vocab)
        self.matrix_segs['tri'] = len(tri_vocab)
        return data_bundle

    def process_from_file(self, paths: str) -> DataBundle:
        loader = ACE2005Loader()
        dl = loader.load(paths)
        return self.process(dl)


class ACE2005Loader(Loader):
    def _load(self, path):
        dataset = DataSet()
        with open(path, 'r') as f:
            is_ere = 'ere' in path.lower()
            for line in f:
                line = line.strip()
                data = json.loads(line)
                tokens = data['tokens']
                if 'event_mentions' in data:
                    event_mentions = data['event_mentions']
                else:
                    event_mentions = data['events']
                ents = {}
                if 'entity_mentions' in data:
                    for ent in data['entity_mentions']:
                        ents[ent['id']] = ent
                else:
                    for ent in data['entities']:
                        ents[ent['entity_id']] = ent
                raw_tris = []
                raw_args = []
                used_ent_ids = []
                for event in event_mentions:
                    e_type = event['event_type']
                    s, e = event['trigger']['start'], event['trigger']['end']
                    assert ''.join(tokens[s:e]) == event['trigger']['text'].replace(' ', '')
                    raw_tris.append((s, e - 1, e_type.lower() + ':' + event['event_subtype'] if is_ere else e_type))
                    _raw_args = []
                    arguments = event['arguments']
                    for arg in arguments:
                        ent_id = arg['entity_id']
                        ent = ents[ent_id]
                        s, e = ent['start'], ent['end'] - 1
                        _raw_args.append([s, e, arg['role'].lower()])
                        used_ent_ids.append(ent_id)
                    raw_args.append(_raw_args)
                for ent_id in used_ent_ids:
                    ent = ents[ent_id]
                    # 修改 ACE2005 train.oneie.json AGGRESSIVEVOICEDAILY_20041208.2133-2
                    assert ''.join(tokens[ent['start']:ent['end']]) == ent['text'].replace(' ', '')

                ins = Instance(raw_words=tokens, raw_tris=raw_tris, raw_args=raw_args, sent_id=data['sent_id'])
                dataset.append(ins)
        return dataset


class OACE2005Loader(Loader):
    # 根据oneie的做法， （1）如果是nested entity的，选择前一个entity；（2）一个role出现在多个事件中，只选择一个
    def _load(self, path):
        dataset = DataSet()
        count = 0
        with open(path, 'r') as f:
            is_ere = 'ere' in path.lower()
            for line in f:
                line = line.strip()
                data = json.loads(line)
                tokens = data['tokens']
                if 'event_mentions' in data:
                    event_mentions = data['event_mentions']
                else:
                    event_mentions = data['events']
                ents = {}
                if 'entity_mentions' in data:
                    for ent in data['entity_mentions']:
                        ents[ent['id']] = ent
                else:
                    for ent in data['entities']:
                        ents[ent['entity_id']] = ent
                raw_tris = []
                raw_args = []
                used_ent_ids = []
                flags = [[] for _ in range(len(tokens))]
                nested_set = set()
                for ent_id, ent in ents.items():
                    for i in range(ent['start'], ent['end']):
                        if flags[i]:
                            for _ent_id in flags[i]:
                                nested_set.add(frozenset([_ent_id, ent_id]))
                        flags[i].append(ent_id)

                for event in event_mentions:
                    e_type = event['event_type']
                    s, e = event['trigger']['start'], event['trigger']['end']
                    assert ''.join(tokens[s:e]) == event['trigger']['text'].replace(' ', '')
                    raw_tris.append((s, e - 1, e_type.lower() + ':' + event['event_subtype'] if is_ere else e_type))
                    _raw_args = []
                    arguments = event['arguments']
                    _ent_ids = set()
                    for arg in arguments:
                        ent_id = arg['entity_id']
                        if ent_id in _ent_ids:
                            count += 1
                            continue
                        _ent_ids.add(ent_id)
                        ent = ents[ent_id]
                        s, e = ent['start'], ent['end'] - 1
                        _raw_args.append([s, e, arg['role'].lower()])
                        used_ent_ids.append(ent_id)
                    raw_args.append(_raw_args)

                # make sure no nested entities is used.
                for ent_id1 in used_ent_ids:
                    for ent_id2 in used_ent_ids:
                        assert frozenset([ent_id1, ent_id2]) not in nested_set

                for ent_id in used_ent_ids:
                    ent = ents[ent_id]
                    # 修改 ACE2005 train.oneie.json AGGRESSIVEVOICEDAILY_20041208.2133-2
                    assert ''.join(tokens[ent['start']:ent['end']]) == ent['text'].replace(' ', '')

                ins = Instance(raw_words=tokens, raw_tris=raw_tris, raw_args=raw_args, sent_id=data['sent_id'])
                dataset.append(ins)
        print(f"Remove {count} roles for {path}")
        return dataset


if __name__ == '__main__':
    # webnlg_pipe = RePipe('roberta-base')
    # dl = webnlg_pipe.process_from_file('/remote-home/hyan01/exps/RelationExtraction/data/tplinker_webnlg')
    paths = '/remote-home/hyan01/exps/RelationExtraction/data/UniRE_SciERC'
    # paths = '/remote-home/hyan01/exps/RelationExtraction/data/UniRE_ace2005'
