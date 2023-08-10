from itertools import chain
import re
from tqdm import tqdm

digit_re = re.compile('\d')

from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle, iob2
from fastNLP.core.metrics.span_f1_pre_rec_metric import _bio_tag_to_spans
import numpy as np
import sparse

from data.pipe import UnifyPipe


class NERDocPipe(UnifyPipe):
    def __init__(self, model_name, max_len=400):
        super().__init__(model_name)
        self.max_len = max_len
        self.matrix_segs = {}  # 用来记录 matrix 最后一维的分别代表啥意思，dict的顺序就是label的顺序，所有value
        # sum起来是该维度大小

    def process(self, data_bundle):
        word2bpes = {}
        labels = set()
        for ins in data_bundle.get_dataset('train'):
            raw_tags = ins['raw_tags']

            for l in chain(*raw_tags):
                assert l.islower()
                if l != 'o':
                    labels.add(l.split('-')[-1])
        labels = list(sorted(labels))
        label2idx = {l: i for i, l in enumerate(labels)}

        def get_new_ins(bpes, spans, indexes):
            bpes.append(self.sep)
            cur_word_idx = indexes[-1]
            indexes.append(0)
            # int8范围-128~127
            matrix = np.zeros((cur_word_idx, cur_word_idx, len(label2idx)), dtype=np.int8)
            ent_target = []
            for _ner in spans:
                s, e, t = _ner
                matrix[s, e, t] = 1
                matrix[e, s, t] = 1
                ent_target.append((s, e, t))
            matrix = sparse.COO.from_numpy(matrix)
            new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                               word_len=cur_word_idx, matrix=matrix, ent_target=ent_target)
            return new_ins

        def process(ins):
            raw_sents = ins['raw_sents']  # [['', ''], ...]
            raw_tags = ins['raw_tags']  # [['', ''], ...]
            bpes = [self.cls]
            indexes = [0]
            spans = []
            ins_lst = []
            for _raw_words, _raw_ents in zip(raw_sents, raw_tags):
                _indexes = []
                _bpes = []
                _spans = _bio_tag_to_spans(_raw_ents)
                _raw_ents = [(s, e - 1, t) for t, (s, e) in _spans]
                for idx, word in enumerate(_raw_words, start=0):
                    word = digit_re.sub('0', word)
                    if word in word2bpes:
                        __bpes = word2bpes[word]
                    else:
                        __bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                       add_special_tokens=False)[:5]
                        word2bpes[word] = __bpes
                    _indexes.extend([idx] * len(__bpes))
                    _bpes.extend(__bpes)
                next_word_idx = indexes[-1] + 1
                if len(bpes) + len(_bpes) <= self.max_len:
                    bpes = bpes + _bpes
                    indexes += [i + next_word_idx for i in _indexes]
                    spans += [(s + next_word_idx - 1, e + next_word_idx - 1, label2idx.get(t),) for s, e, t in
                              _raw_ents]
                else:
                    new_ins = get_new_ins(bpes, spans, indexes)
                    ins_lst.append(new_ins)
                    indexes = [0] + [i + 1 for i in _indexes]
                    spans = [(s, e, label2idx.get(t),) for s, e, t in _raw_ents]
                    bpes = [self.cls] + _bpes
            if bpes:
                ins_lst.append(get_new_ins(bpes, spans, indexes))

            return ins_lst

        for name in data_bundle.get_dataset_names():
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            for ins in tqdm(ds, total=len(ds), desc=name, leave=False):
                ins_lst = process(ins)
                for ins in ins_lst:
                    new_ds.append(ins)
            data_bundle.set_dataset(new_ds, name)
        setattr(data_bundle, 'label2idx', label2idx)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('ent_target', None)
        self.matrix_segs['ent'] = len(label2idx)
        return data_bundle

    def process_from_file(self, paths: str) -> DataBundle:
        if isinstance(paths, dict):
            path = list(paths.values())[0]
        else:
            path = paths
        if 'conll2003' in path:
            dl = Conll2003NERDocLoader().load(paths)
        elif 'ontonotes' in path:
            dl = OntoNotesNERDocLoader().load(paths)
        return self.process(dl)


class Conll2003NERDocLoader(Loader):
    """
    读取之后为
    "raw_sents":
       [[word1, word2, word3, ...], [word1, word2], ...]
    "raw_tags":
       [[t1, t2, ..], [t1, t2]]

    """

    def __init__(self):
        super().__init__()

    def _load(self, path):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            sents = []
            tags = []
            sent = []
            tag = []
            for line in f:
                if line.startswith('-DOCSTART-'):
                    if sents:  # 说明上一句完了
                        ins = Instance(raw_sents=sents, raw_tags=tags)
                        ds.append(ins)
                        sents = []
                        tags = []
                        sent = []
                        tag = []
                elif line.strip() == '':  # 一句话完了
                    if sent:
                        sents.append(sent)
                        tags.append([_.lower() for _ in iob2(tag)])
                        sent = []
                        tag = []
                else:
                    line = line.strip()
                    parts = line.split()
                    sent.append(parts[0])
                    tag.append(parts[-1])

            if sents:  # 最后一句话需要处理
                ins = Instance(raw_sents=sents, raw_tags=tags)
                ds.append(ins)

        return ds


class OntoNotesNERDocLoader(Loader):
    """
    用以读取OntoNotes的NER数据，同时也是Conll2012的NER任务数据。将OntoNote数据处理为conll格式的过程可以参考
    https://github.com/yhcc/OntoNotes-5.0-NER。OntoNoteNERLoader将取第4列和第11列的内容。

    读取的数据格式为：

    Example::

        bc/msnbc/00/msnbc_0000   0   0          Hi   UH   (TOP(FRAG(INTJ*)  -   -   -    Dan_Abrams  *   -
        bc/msnbc/00/msnbc_0000   0   1    everyone   NN              (NP*)  -   -   -    Dan_Abrams  *   -
        ...

    返回的DataSet的内容为

    读取之后为
    "raw_sents":
       [[word1, word2, word3, ...], [word1, word2], ...]
    "raw_tags":
       [[t1, t2, ..], [t1, t2]]

    """

    def __init__(self):
        super().__init__()

    def _load(self, path: str):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            cur_part_id = 0
            sents = []
            tags = []
            sent = []
            tag = []
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    doc_id, part_id, word, en = parts[0], parts[1], parts[3], parts[10]
                    part_id = doc_id + ' ' + part_id
                    if part_id != cur_part_id:
                        if sents:
                            ds.append(Instance(sents=sents, tags=tags))
                            sents = []
                            tags = []
                            sent = []
                            tag = []
                        cur_part_id = part_id
                    sent.append(word)
                    tag.append(en)
                else:
                    if sent:
                        sents.append(sent)
                        tags.append(tag)
                        sent = []
                        tag = []
            if sent:
                sents.append(sent)
                tags.append(tag)
            if sents:
                ds.append(Instance(sents=sents, tags=tags))

        def convert_word(sents):
            converted_sents = []
            for words in sents:
                converted_words = []
                for word in words:
                    word = word.replace('/.', '.')  # 有些结尾的.是/.形式的
                    if not word.startswith('-'):
                        converted_words.append(word)
                        continue
                    # 以下是由于这些符号被转义了，再转回来
                    tfrs = {'-LRB-': '(',
                            '-RRB-': ')',
                            '-LSB-': '[',
                            '-RSB-': ']',
                            '-LCB-': '{',
                            '-RCB-': '}'
                            }
                    if word in tfrs:
                        converted_words.append(tfrs[word])
                    else:
                        converted_words.append(word)
                converted_sents.append(converted_words)
            return converted_sents

        def convert_to_bio(tagss):
            flag = None
            bio_tagss = []
            for tags in tagss:
                bio_tags = []
                for tag in tags:
                    label = tag.strip("()*")
                    if '(' in tag:
                        bio_label = 'B-' + label
                        flag = label
                    elif flag:
                        bio_label = 'I-' + flag
                    else:
                        bio_label = 'O'
                    if ')' in tag:
                        flag = None
                    bio_tags.append(bio_label.lower())
                bio_tagss.append(bio_tags)
            return bio_tagss

        ds.apply_field(convert_word, field_name='sents', new_field_name='raw_sents')
        ds.apply_field(convert_to_bio, field_name='tags', new_field_name='raw_tags')

        return ds

    def download(self):
        raise RuntimeError("Ontonotes cannot be downloaded automatically, you can refer "
                           "https://github.com/yhcc/OntoNotes-5.0-NER to download and preprocess.")
