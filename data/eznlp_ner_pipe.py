import json
import re

digit_re = re.compile('\d')

from fastNLP import DataSet, Instance
from fastNLP.io import DataBundle, Loader
from fastNLP import Vocabulary
import numpy as np
import sparse

from data.pipe import UnifyPipe


class eznlpLoader(Loader):
    def _load(self, path):
        ds = DataSet()
        with open(path, 'r') as f:
            data = json.load(f)
        for d in data:
            ins = Instance(**d)
            ds.append(ins)
        return ds


class eznlpDocPipe(UnifyPipe):
    # ent的分数不再是需要一个square了
    def __init__(self, model_name):
        super().__init__(model_name)
        self.matrix_segs = {}

    def process(self, data_bundle: DataBundle) -> DataBundle:
        word2bpes = {}
        ner_vocab = Vocabulary(padding=None, unknown=None)

        for ins in data_bundle.get_dataset('train'):
            for t in ins['chunks']:
                ner_vocab.add(t[0])
        self.matrix_segs['ent'] = len(ner_vocab)
        word2bpes = {}

        def process(ins):
            raw_words = ins['tokens']  # [['', ''], ...]
            chunks = ins['chunks']  # [['', ''], ...]
            # 初始化的
            bpes = [self.tokenizer.cls_token_id]
            indexes = [0]
            for idx, word in enumerate(raw_words, start=1):
                if len(word) > 1 and word[0] == '/':
                    word = word[1:]
                if word in word2bpes:
                    __bpes = word2bpes[word]
                else:
                    __bpes = self.tokenizer.encode(' ' + word if self.add_prefix_space else word,
                                                   add_special_tokens=False)[:5]
                    word2bpes[word] = __bpes
                indexes.extend([idx] * len(__bpes))
                bpes.extend(__bpes)
            bpes.append(self.tokenizer.sep_token_id)
            indexes.append(0)
            ent_matrix = np.zeros((len(raw_words), len(raw_words), len(ner_vocab)), dtype=bool)
            ner_target = []
            for _ner in chunks:
                t, s, e = _ner
                s, e = s, e - 1
                ent_matrix[s, e, ner_vocab.to_index(t)] = 1
                ent_matrix[e, s, ner_vocab.to_index(t)] = 1
                ner_target.append((s, e, ner_vocab.to_index(t)))
            ent_matrix = sparse.COO.from_numpy(ent_matrix)
            new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                               word_len=len(raw_words), matrix=ent_matrix, ent_target=ner_target)
            return new_ins

        data_bundle.apply_more(process, num_proc=4, progress_bar='tqdm', progress_desc='process')
        setattr(data_bundle, 'ner_vocab', ner_vocab)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('ent_target', None)
        return data_bundle

    def process_from_file(self, paths: str):
        loader = eznlpLoader()

        dl = loader.load(paths)
        return self.process(dl)
