import collections
from collections import defaultdict
import json

from fastNLP import Vocabulary
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
import numpy as np
import sparse

from data.pipe import UnifyPipe


class OneIEPipe(UnifyPipe):
    def __init__(self, model_name):
        super(OneIEPipe, self).__init__(model_name=model_name)
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

        ent_vocab = Vocabulary(padding=None, unknown=None)
        rel_vocab = Vocabulary(padding=None, unknown=None)

        for name, ds in data_bundle.iter_datasets():
            for ins in ds:
                ent_vocab.add_word_lst([_t[-1] for _t in ins['raw_ents']])
                rel_vocab.add_word_lst([_t[-1] for _t in ins['raw_rels']])

        # TODO entity

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

                matrix = np.zeros(
                    (len(sent), len(sent), len(ent_vocab) + len(rel_vocab) + 1 + len(tri_vocab) + len(role_vocab)),
                    dtype=np.int8)
                num_ent = len(ent_vocab)
                num_rel = len(rel_vocab)

                ent_target = []
                ner_dict = set()
                ent_mask = [0] * len(sent)
                for _ner in ins['raw_ents']:
                    s, e, t = _ner
                    overlapped = 0
                    for i in range(s, e + 1):  # 没有overlap的
                        if ent_mask[i] != 0:
                            overlapped += 1
                        ent_mask[i] = 1
                    if overlapped:
                        print("Overlapped")
                    matrix[s, e, ent_vocab.to_index(t)] = 1
                    matrix[e, s, ent_vocab.to_index(t)] = 1
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
                            rel_mask[s2, s1] = 1
                            rel_mask[e2, e1] = 1

                rel_target = set()
                shift = len(ent_vocab)
                for _rel in set([tuple(_rel) for _rel in ins['raw_rels']]):
                    s1, e1, s2, e2, r = _rel
                    assert (s1, e1) in ner_dict and (s2, e2) in ner_dict
                    assert matrix[s1, e1].any() == 1 and matrix[s2, e2].any() == 1
                    r_idx = rel_vocab.to_index(r)
                    matrix[s1, s2, r_idx + shift] = 1
                    matrix[e1, e2, r_idx + shift] = 1
                    # todo 这里是由于oneie的代码中不考虑relation的方向，所以我们这里也不考虑了
                    matrix[s2, s1, r_idx + shift] = 1
                    matrix[e2, e1, r_idx + shift] = 1
                    rel_target.add((s1, e1, s2, e2, r_idx))
                rel_target = list(rel_target)

                tri_target = []
                tri_mask = [0] * len(sent)
                for tri in raw_tris:
                    s, e, t = tri
                    for i in range(s, e + 1):  # 没有overlap的
                        assert tri_mask[i] == 0
                        tri_mask[i] = 1
                    assert matrix[s, e, tri_vocab.to_index(t) + 1 + num_rel + num_ent] == 0  # trigger 不会overlap
                    matrix[s, e, tri_vocab.to_index(t) + 1 + num_rel + num_ent] = 1
                    matrix[e, s, tri_vocab.to_index(t) + 1 + num_ent + num_rel] = 1
                    tri_target.append((s, e, tri_vocab.to_index(t)))

                arg_target = []
                roles = []
                checker = collections.Counter()
                for tri, args in zip(raw_tris, raw_args):
                    s, e, t = tri
                    t = tri_vocab.to_index(t)
                    for arg in args:
                        a_s, a_e, a = arg
                        a = role_vocab.to_index(a) + len(tri_vocab) + 1 + num_ent + num_rel
                        matrix[a_s, a_e, num_rel + num_ent] = 1
                        matrix[a_e, a_s, num_rel + num_ent] = 1
                        matrix[s, a_s, a] = 1
                        matrix[e, a_e, a] = 1
                        # 因为可以是对称的？
                        matrix[a_s, s, a] = 1
                        matrix[a_e, e, a] = 1
                        constrain[t].add(a - 1 - len(tri_vocab) - (num_ent + num_rel))
                        roles.append((a_s, a_e))
                        arg_target.append((a_s, a_e, a - 1 - len(tri_vocab) - (num_ent + num_rel), t))
                        checker[arg_target[-1]] += 1

                # 用来给没有关系地方进行采样处理的
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
                        # _tmp = [s1, e1, s2, e2]
                        # for x in range(4):
                        #     for y in range(4):
                        #         if x!=y:
                        #             rel_mask[_tmp[x], _tmp[y]] = 1
                matrix = sparse.COO.from_numpy(matrix)
                rel_mask = sparse.COO.from_numpy(rel_mask)
                new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                                   word_len=len(sent), matrix=matrix, tri_target=tri_target,
                                   arg_target=arg_target, rel_mask=rel_mask, ent_target=ent_target,
                                   rel_target=rel_target, tokens=sent)
                new_ds.append(new_ins)
            data_bundle.set_dataset(new_ds, name=name)
        setattr(data_bundle, 'tri_vocab', tri_vocab)
        setattr(data_bundle, 'role_vocab', role_vocab)
        setattr(data_bundle, 'ent_vocab', ent_vocab)
        setattr(data_bundle, 'rel_vocab', rel_vocab)
        setattr(data_bundle, 'constrain', constrain)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('tri_target', None)
        data_bundle.set_pad('arg_target', None)
        data_bundle.set_pad('ent_target', None)
        data_bundle.set_pad('rel_target', None)
        data_bundle.set_pad('tokens', None)
        self.matrix_segs['ent'] = len(ent_vocab)
        self.matrix_segs['rel'] = len(rel_vocab)
        self.matrix_segs['arg'] = 1
        self.matrix_segs['role'] = len(role_vocab)
        self.matrix_segs['tri'] = len(tri_vocab)
        return data_bundle

    def process_from_file(self, paths: str) -> DataBundle:
        loader = OneIELoader()
        dl = loader.load(paths)
        return self.process(dl)


def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        break_flag = False
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]
                break_flag = True
        if break_flag:
            continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map


class OneIELoader(Loader):
    def _load(self, path):
        dataset = DataSet()
        is_ere = False
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                data = json.loads(line)
                tokens = data['tokens']
                if len(tokens) == 0 or ('sentence' in data and data['sentence'].strip() == ''):
                    continue
                event_mentions = data['event_mentions'] if 'event_mentions' in data else data['events']
                relation_mentions = data['relation_mentions'] if 'relation_mentions' in data else data['relations']
                entity_mentions = data['entity_mentions'] if 'entity_mentions' in data else data['entities']
                ents = {}
                raw_tris = []
                raw_args = []
                raw_ents = []
                raw_rels = []

                for ent in entity_mentions:
                    # pure_text = re.sub(' ', '', ent['text'])
                    # if ''.join(tokens[ent['start']:ent['end']])!=pure_text:
                    #     while ''.join(tokens[ent['start']:ent['end']]) in pure_text and ent['end']<len(tokens) and \
                    #             ''.join(tokens[ent['start']:ent['end']]) != pure_text:
                    #         ent['end'] += 1
                    # assert ''.join(tokens[ent['start']:ent['end']]) == pure_text, (' '.join(tokens[ent['start']:ent['end']]), ent['text'])
                    raw_ents.append((ent['start'], ent['end'] - 1, ent['entity_type']))
                    ents[ent['id'] if 'id' in ent else ent['entity_id']] = ent

                for rel in relation_mentions:
                    if 'arguments' in rel:
                        ent1, ent2 = ents[rel['arguments'][0]['entity_id']], ents[rel['arguments'][1]['entity_id']]
                    else:
                        ent1, ent2 = ents[rel['arg1']['entity_id']], ents[rel['arg2']['entity_id']]

                    raw_rels.append(
                        (ent1['start'], ent1['end'] - 1, ent2['start'], ent2['end'] - 1, rel['relation_type']))

                for event in event_mentions:
                    e_type = event['event_type'] + ':' + event['event_subtype'] if is_ere else event['event_type']
                    s, e = event['trigger']['start'], event['trigger']['end']
                    assert ''.join(tokens[s:e]) == event['trigger']['text'].replace(' ', '')
                    raw_tris.append((s, e - 1, e_type))
                    _raw_args = []
                    arguments = event['arguments']
                    for arg in arguments:
                        ent_id = arg['entity_id']
                        ent = ents[ent_id]
                        s, e = ent['start'], ent['end'] - 1
                        _raw_args.append([s, e, arg['role'].lower()])
                    raw_args.append(_raw_args)

                ins = Instance(raw_words=tokens, raw_tris=raw_tris, raw_args=raw_args, raw_ents=raw_ents,
                               raw_rels=raw_rels)
                dataset.append(ins)
        return dataset


class FollowOneIEPipe(UnifyPipe):
    def __init__(self, model_name):
        super(FollowOneIEPipe, self).__init__(model_name=model_name)
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

        ent_vocab = Vocabulary(padding=None, unknown=None)
        rel_vocab = Vocabulary(padding=None, unknown=None)

        for name, ds in data_bundle.iter_datasets():
            for ins in ds:
                ent_vocab.add_word_lst([_t[-1] for _t in ins['raw_ents']])
                rel_vocab.add_word_lst([_t[-1] for _t in ins['raw_rels']])

        # constrain: {key: set()}  其中key为event_type的index，value是这个event_type运行的role有哪些
        constrain = defaultdict(set)  #
        for name in data_bundle.get_dataset_names():
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            overlong_num = 0
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
                # if len(bpes)>300 and name == 'train':
                #     print(f"Skip for {name} because {len(bpes)} length")
                #     continue
                if len(bpes) > 128:  # follow oneie
                    overlong_num += 1
                    continue
                assert len(bpes) <= 512
                indexes.append(0)

                matrix = np.zeros(
                    (len(sent), len(sent), len(ent_vocab) + len(rel_vocab) + 1 + len(tri_vocab) + len(role_vocab)),
                    dtype=np.int8)
                num_ent = len(ent_vocab)
                num_rel = len(rel_vocab)

                ent_target = []
                ner_dict = set()
                ent_mask = [0] * len(sent)
                for _ner in ins['raw_ents']:
                    s, e, t = _ner
                    overlapped = 0
                    for i in range(s, e + 1):  # 没有overlap的
                        if ent_mask[i] != 0:
                            overlapped += 1
                        ent_mask[i] = 1
                    if overlapped:
                        print("Overlapped")
                    matrix[s, e, ent_vocab.to_index(t)] = 1
                    matrix[e, s, ent_vocab.to_index(t)] = 1
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
                            rel_mask[s2, s1] = 1
                            rel_mask[e2, e1] = 1

                rel_target = set()
                shift = len(ent_vocab)
                for _rel in set([tuple(_rel) for _rel in ins['raw_rels']]):
                    s1, e1, s2, e2, r = _rel
                    assert (s1, e1) in ner_dict and (s2, e2) in ner_dict
                    assert matrix[s1, e1].any() == 1 and matrix[s2, e2].any() == 1
                    r_idx = rel_vocab.to_index(r)
                    matrix[s1, s2, r_idx + shift] = 1
                    matrix[e1, e2, r_idx + shift] = 1
                    # todo 这里是由于oneie的代码中不考虑relation的方向，所以我们这里也不考虑了
                    matrix[s2, s1, r_idx + shift] = 1
                    matrix[e2, e1, r_idx + shift] = 1
                    rel_target.add((s1, e1, s2, e2, r_idx))
                rel_target = list(rel_target)

                tri_target = []
                tri_mask = [0] * len(sent)
                for tri in raw_tris:
                    s, e, t = tri
                    for i in range(s, e + 1):  # 没有overlap的
                        assert tri_mask[i] == 0
                        tri_mask[i] = 1
                    assert matrix[s, e, tri_vocab.to_index(t) + 1 + num_rel + num_ent] == 0  # trigger 不会overlap
                    matrix[s, e, tri_vocab.to_index(t) + 1 + num_rel + num_ent] = 1
                    matrix[e, s, tri_vocab.to_index(t) + 1 + num_ent + num_rel] = 1
                    tri_target.append((s, e, tri_vocab.to_index(t)))

                arg_target = []
                roles = []
                checker = collections.Counter()
                for tri, args in zip(raw_tris, raw_args):
                    s, e, t = tri
                    t = tri_vocab.to_index(t)
                    for arg in args:
                        a_s, a_e, a = arg
                        a = role_vocab.to_index(a) + len(tri_vocab) + 1 + num_ent + num_rel
                        matrix[a_s, a_e, num_rel + num_ent] = 1
                        matrix[a_e, a_s, num_rel + num_ent] = 1
                        matrix[s, a_s, a] = 1
                        matrix[e, a_e, a] = 1
                        # 因为可以是对称的？
                        matrix[a_s, s, a] = 1
                        matrix[a_e, e, a] = 1
                        constrain[t].add(a - 1 - len(tri_vocab) - (num_ent + num_rel))
                        roles.append((a_s, a_e))
                        arg_target.append((a_s, a_e, a - 1 - len(tri_vocab) - (num_ent + num_rel), t))
                        checker[arg_target[-1]] += 1

                # 用来给没有关系地方进行采样处理的
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
                                   arg_target=arg_target, rel_mask=rel_mask, ent_target=ent_target,
                                   rel_target=rel_target, tokens=sent)
                new_ds.append(new_ins)
            print(f"Delete {overlong_num} sentences for overlong...")
            data_bundle.set_dataset(new_ds, name=name)
        setattr(data_bundle, 'tri_vocab', tri_vocab)
        setattr(data_bundle, 'role_vocab', role_vocab)
        setattr(data_bundle, 'ent_vocab', ent_vocab)
        setattr(data_bundle, 'rel_vocab', rel_vocab)
        setattr(data_bundle, 'constrain', constrain)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('tri_target', None)
        data_bundle.set_pad('arg_target', None)
        data_bundle.set_pad('ent_target', None)
        data_bundle.set_pad('rel_target', None)
        data_bundle.set_pad('tokens', None)
        self.matrix_segs['ent'] = len(ent_vocab)
        self.matrix_segs['rel'] = len(rel_vocab)
        self.matrix_segs['arg'] = 1
        self.matrix_segs['role'] = len(role_vocab)
        self.matrix_segs['tri'] = len(tri_vocab)
        return data_bundle

    def process_from_file(self, paths: str) -> DataBundle:
        loader = FollowOneIELoader()
        dl = loader.load(paths)
        return self.process(dl)


class FollowOneIELoader(Loader):
    """
    （1）删掉title
    （2）删掉overlap的entity, 但这个在数据中好像就被干掉了
    （3）删除重复的role
    """

    def _load(self, path):
        dataset = DataSet()
        is_ere = False
        delete_title_count = 0
        count = 0
        duplicate_count = 0
        delete_ent = 0
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                data = json.loads(line)
                tokens = data['tokens']
                if len(tokens) == 0 or ('sentence' in data and data['sentence'].strip() == ''):
                    continue
                event_mentions = data['event_mentions'] if 'event_mentions' in data else data['events']
                relation_mentions = data['relation_mentions'] if 'relation_mentions' in data else data['relations']
                entity_mentions = data['entity_mentions'] if 'entity_mentions' in data else data['entities']
                ents = {}
                raw_tris = []
                raw_args = []
                raw_ents = []
                raw_rels = []

                _entity_mentions, ent_id_map = remove_overlap_entities(entity_mentions)
                delete_ent += len(entity_mentions) - len(_entity_mentions)
                entity_mentions = _entity_mentions
                for ent in entity_mentions:
                    # pure_text = re.sub(' ', '', ent['text'])
                    # if ''.join(tokens[ent['start']:ent['end']])!=pure_text:
                    #     while ''.join(tokens[ent['start']:ent['end']]) in pure_text and ent['end']<len(tokens) and \
                    #             ''.join(tokens[ent['start']:ent['end']]) != pure_text:
                    #         ent['end'] += 1
                    # assert ''.join(tokens[ent['start']:ent['end']]) == pure_text, (' '.join(tokens[ent['start']:ent['end']]), ent['text'])
                    raw_ents.append((ent['start'], ent['end'] - 1, ent['entity_type']))
                    ents[ent['id'] if 'id' in ent else ent['entity_id']] = ent
                seen_rels = set()
                for rel in relation_mentions:
                    if 'arguments' in rel:
                        ent_id1 = ent_id_map.get(rel['arguments'][0]['entity_id'], rel['arguments'][0]['entity_id'])
                        ent_id2 = ent_id_map.get(rel['arguments'][1]['entity_id'], rel['arguments'][1]['entity_id'])
                    else:
                        ent_id1 = ent_id_map.get(rel['arg1']['entity_id'], rel['arg1']['entity_id'])
                        ent_id2 = ent_id_map.get(rel['arg2']['entity_id'], rel['arg2']['entity_id'])
                    ent1, ent2 = ents[ent_id1], ents[ent_id2]
                    if frozenset([ent_id1, ent_id2]) in seen_rels:
                        duplicate_count += 1
                        continue
                    seen_rels.add(frozenset([ent_id1, ent_id2]))
                    raw_rels.append(
                        (ent1['start'], ent1['end'] - 1, ent2['start'], ent2['end'] - 1, rel['relation_type']))

                for event in event_mentions:
                    e_type = event['event_type'] + ':' + event['event_subtype'] if is_ere else event['event_type']
                    s, e = event['trigger']['start'], event['trigger']['end']
                    assert ''.join(tokens[s:e]) == event['trigger']['text'].replace(' ', '')
                    raw_tris.append((s, e - 1, e_type))
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

                    raw_args.append(_raw_args)

                ins = Instance(raw_words=tokens, raw_tris=raw_tris, raw_args=raw_args, raw_ents=raw_ents,
                               raw_rels=raw_rels, sent_id=data['sent_id'])
                dataset.append(ins)
        # print(f"Delete {delete_title_count} sentences, because it is title.")
        print(f"Remove {count} roles for {path}")
        print(f'Delete duplicated {duplicate_count} relations for {path}')
        print(f"Remove {delete_ent} entities for {path}")
        return dataset


"""
oneie中的特殊处理
（1）去掉overlapped entity；但这个其实如果是通过使用它的process代码来的话，是本身就去掉了
（2）同一个trigger中，一个span只能是某一种role
（3）同一组entity pair评测的时候只能有一种关系
（4）entity的关系评测的时候，只选择某一个方向进行评测，其实都是靠前的entity在前面，
    评测的时候也是只有一个的

"""
