from fastNLP import Vocabulary, Instance
from fastNLP.core.metrics.span_f1_pre_rec_metric import _bio_tag_to_spans
from fastNLP.io import OntoNotesNERLoader
from fastNLP.io import Pipe, DataBundle, Conll2003NERLoader, iob2
import numpy as np
import sparse
from transformers import AutoTokenizer


class ConllPipe(Pipe):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.add_space = True if 'roberta' in model_name.lower() else False
        self.matrix_segs = {}

    def process(self, data_bundle: DataBundle) -> DataBundle:
        word2bpes = {}
        ner_vocab = Vocabulary(padding=None, unknown=None)
        data_bundle.apply_field(iob2, field_name='target', new_field_name='target', num_proc=4)

        def get_span(target):
            spans = _bio_tag_to_spans(target)
            return spans

        data_bundle.apply_field(get_span, field_name='target', new_field_name='spans', num_proc=4, progress_desc='span')
        for ins in data_bundle.get_dataset('train'):
            ner_vocab.add_word_lst([s[0] for s in ins['spans']])
        self.matrix_segs['ent'] = len(ner_vocab)
        word2bpes = {}

        def process(ins):
            raw_words = ins['raw_words']
            spans = ins['spans']
            bpes = [self.tokenizer.cls_token_id]
            indexes = [0]
            for idx, word in enumerate(raw_words, start=1):
                # word = digit_re.sub('0', word)
                if word in word2bpes:
                    _bpes = word2bpes[word]
                else:
                    _bpes = self.tokenizer.encode(' ' + word if self.add_space else word, add_special_tokens=False)[:5]
                    word2bpes[word] = _bpes
                indexes.extend([idx] * len(_bpes))
                bpes.extend(_bpes)
            bpes.append(self.tokenizer.sep_token_id)
            indexes.append(0)
            ent_matrix = np.zeros((len(raw_words), len(raw_words), len(ner_vocab)), dtype=bool)
            ner_target = []
            for _ner in spans:
                t, (s, e) = _ner
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
        dataset = ''
        if isinstance(paths, dict):
            dataset = 'conll2003'
        elif 'ontonotes' in paths:
            dataset = 'ontonotes'
        if dataset == 'conll2003':
            loader = Conll2003NERLoader()
        elif dataset == 'ontonotes':
            loader = OntoNotesNERLoader()

        dl = loader.load(paths)
        return self.process(dl)
