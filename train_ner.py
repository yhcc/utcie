import os
import sys

sys.path.append('..')

if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings

warnings.filterwarnings('ignore')

import argparse
import numpy as np

import fastNLP
from fastNLP import cache_results, prepare_torch_dataloader, print
from fastNLP import Trainer, Evaluator
from fastNLP import TorchGradClipCallback
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP.core.dataloaders.utils import OverfitDataLoader
import fitlog
import torch

from data.flat_doc_pipe import NERDocPipe
from data.flat_sent_pipe import ConllPipe
from data.eznlp_ner_pipe import eznlpDocPipe
from data.nest_ner_pipe import SentACENerPipe, GeniaNerPipe
from data.padder import Torch3DMatrixPadder
from model.args import ARGS
from model.batch_sampler import ConstantTokenBatchSampler
from model.callbacks import FitlogCallback, TorchWarmupCallback
from model.ner_metrics import NERMetric, FastNERMetric
from model.unify_model import UnifyModel

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=3e-5, type=float)
parser.add_argument('-b', '--batch_size', default=48, type=int)
parser.add_argument('-n', '--n_epochs', default=30, type=int)
parser.add_argument('-a', '--accumulation_steps', default=1, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='conll2003_', type=str)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--attn_dropout', default=0.15, type=float)
parser.add_argument('--cross_depth', default=3, type=int)
parser.add_argument('--cross_dim', default=32, type=int)
parser.add_argument('--use_ln', default=1, type=int)
parser.add_argument('--use_s2', default=1, type=int)
parser.add_argument('--use_gelu', default=0, type=int)
parser.add_argument('--drop_s1_p', default=0, type=float)
parser.add_argument('--empty_rel_weight', default=0.1, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--ent_thres', default=0.5, type=float)  # 没有这个参数的的是0.5
parser.add_argument('--use_tri_bias', default=True, type=int)  # 没有这个值的记录，为False, 这个用来控制选择entity的时候是否选择 上下三角相加
parser.add_argument('--use_size_embed', default=False, type=int)  # 没有这个值的记录，为False, 这个用来控制选择entity的时候是否选择 上下三角相加

args = parser.parse_args()
dataset_name = args.dataset_name
if args.model_name is None:
    if 'genia' in args.dataset_name:
        args.model_name = 'dmis-lab/biobert-v1.1'
    else:
        args.model_name = 'roberta-base'

model_name = args.model_name
ent_thres = args.ent_thres
if args.use_tri_bias:
    args.use_tri_bias = True
else:
    args.use_tri_bias = False
use_tri_bias = args.use_tri_bias
######hyper
non_ptm_lr_ratio = 100
schedule = 'linear'  # 没有这个参数的为linear
weight_decay = 1e-2  # 没有的是1e-2
symmetric = True  # 没有这个参数是False
######hyper
ARGS['use_pos'] = False
ARGS['use_gelu'] = args.use_gelu
ARGS['s1_scale_plus'] = False
ARGS['use_size_embed'] = args.use_size_embed
ARGS['size_embed_dim'] = 100

# 如果是debug模式，就不同步到fitlog
fitlog.debug()

# 这个似乎必须是True，否则收敛起来太慢了。
eval_batch_size = min(args.batch_size * 2, 32)
if model_name == 'microsoft/deberta-v3-base' and dataset_name == 'ace2005':
    eval_batch_size = 18
if 'SEARCH_ID' in os.environ and False:
    fitlog.set_log_dir('debug_flat_logs/')
else:
    fitlog.set_log_dir('debug_flat_logs/')
seed = fitlog.set_rng_seed()
os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
if 'SEARCH_ID' not in os.environ and fastNLP.get_global_rank() == 0:
    fitlog.commit(__file__)
fitlog.add_hyper(args)
fitlog.add_hyper(ARGS)
fitlog.add_hyper_in_file(__file__)
if 'SEARCH_ID' in os.environ:
    fitlog.add_other(os.environ['SEARCH_ID'], name='SEARCH_ID')
    fit_id = fitlog.get_fit_id(__file__)[:8]
    if fit_id is not None:
        fitlog.add_other(name='fit_id', value=fit_id)

fitlog.add_other(name='info', value='biaffine_gelu+2ul_pos')


@cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name):
    if dataset_name in ('conll2003', 'conll2003_'):  # conll2003_是concat了train/dev的
        pipe = ConllPipe(model_name=model_name)
        paths = {"train": "../dataset/conll2003/train.txt",
                 "dev": "../dataset/conll2003/testa.txt",
                 'test': "../dataset/conll2003/test.txt"}
    elif dataset_name == 'ontonotes':
        pipe = ConllPipe(model_name=model_name)
        paths = '../dataset/en-ontonotes/english'
    if dataset_name in ('dconll2003',):  # conll2003_是concat了train/dev的
        pipe = NERDocPipe(model_name=model_name, max_len=400)
        paths = {"train": "../dataset/conll2003/train.txt",
                 "dev": "../dataset/conll2003/testa.txt",
                 'test': "../dataset/conll2003/test.txt"}
    elif dataset_name == 'dontonotes':
        pipe = NERDocPipe(model_name=model_name, max_len=400)
        paths = '../dataset/en-ontonotes/english'
    if dataset_name in ('eznlp_conll2003',):  # conll2003_是concat了train/dev的
        pipe = eznlpDocPipe(model_name=model_name)
        paths = '/remote-home/hyan01/exps/TransformerNER/others/eznlp/save_conll2003'
    elif dataset_name == 'eznlp_ontonotes':
        pipe = eznlpDocPipe(model_name=model_name)
        paths = '/remote-home/hyan01/exps/TransformerNER/others/eznlp/save_ontonotes'
    elif dataset_name == 'ace2004':
        pipe = SentACENerPipe(model_name=model_name)
        paths = '../dataset/en_ace04'
    elif dataset_name == 'ace2005':
        pipe = SentACENerPipe(model_name=model_name)
        paths = '../dataset/en_ace05'
    elif dataset_name == 'genia':
        pipe = GeniaNerPipe(model_name=model_name)
        paths = '../dataset/genia_w2ner'
    elif dataset_name == 'ace2004_new':
        pipe = SentACENerPipe(model_name=model_name)
        paths = '../../TransformerNER/cnn_nested_ner/preprocess/outputs/ace2004'
    elif dataset_name == 'ace2005_new':
        pipe = SentACENerPipe(model_name=model_name)
        paths = '../../TransformerNER/cnn_nested_ner/preprocess/outputs/ace2005'
    elif dataset_name == 'genia_v4':
        pipe = SentACENerPipe(model_name=model_name)
        paths = '../../TransformerNER/cnn_nested_ner/preprocess/outputs/genia'
    dl = pipe.process_from_file(paths)

    return dl, pipe.matrix_segs


dl, matrix_segs = get_data(dataset_name, model_name)

if dataset_name == 'conll2003_':
    dev = dl.get_dataset('dev')
    dl.delete_dataset('dev')
    dl.get_dataset('train').concat(dev)


def densify(x):
    x = x.todense()
    x = x.astype(np.float32)
    return x


dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')
dl.apply_field(lambda x: x.todense().astype(bool), field_name='rel_mask', new_field_name='rel_mask',
               progress_bar='Densify')

print(dl)
label2idx = getattr(dl, 'ner_vocab') if hasattr(dl, 'ner_vocab') else getattr(dl, 'label2idx')
print(f"{len(label2idx)} labels: {label2idx}, matrix_segs:{matrix_segs}")
dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=sum(matrix_segs.values()),
                                                    batch_size=max(32, args.batch_size)))

    if name == 'train':
        if dataset_name in ('ace2004_new', 'ace2005_new', 'genia_v4') or torch.cuda.mem_get_info()[
            0] > 30_043_904_512 or \
                'conll2003' in dataset_name or 'ontonotes' in dataset_name:
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=4,
                                           batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                              batch_size=args.batch_size,
                                                                              num_batch_per_bucket=30),
                                           pin_memory=True, shuffle=True)
        else:
            max_token = 2048
            if 'deberta-v3-base' in model_name and dataset_name == 'ace2005':
                max_token = 1024
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=4,
                                           batch_sampler=ConstantTokenBatchSampler(ds.get_field('bpe_len').content,
                                                                                   max_token=max_token,
                                                                                   max_sentence=args.batch_size,
                                                                                   batch_size_when_max_len=min(
                                                                                       {'ace2004': 8,
                                                                                        'ace2005': 6, 'ace2005_new': 8,
                                                                                        'ace2004_new': 8,
                                                                                        'genia_new': 24,
                                                                                        'ontonotes': 12,
                                                                                        'dontonotes': 6,
                                                                                        'genia': 24}[dataset_name],
                                                                                       args.batch_size)),
                                           pin_memory=True, shuffle=True)
    else:
        _dl = prepare_torch_dataloader(ds, batch_size=eval_batch_size, num_workers=3,
                                       sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
        _dl = OverfitDataLoader(_dl, overfit_batches=-1)
    dls[name] = _dl

model = UnifyModel(model_name, matrix_segs, use_at_loss=False, cross_dim=args.cross_dim,
                   cross_depth=args.cross_depth, biaffine_size=args.biaffine_size, use_ln=args.use_ln,
                   drop_s1_p=args.drop_s1_p, use_s2=args.use_s2, empty_rel_weight=args.empty_rel_weight,
                   attn_dropout=args.attn_dropout, use_tri_bias=use_tri_bias)

# optimizer
parameters = []
ln_params = []
non_ln_params = []
non_pretrain_params = []
non_pretrain_ln_params = []

for name, param in model.named_parameters():
    name = name.lower()
    if param.requires_grad is False:
        continue
    if 'pretrain_model' in name:
        if 'norm' in name or 'bias' in name:
            ln_params.append(param)
        else:
            non_ln_params.append(param)
    else:
        if 'norm' in name or 'bias' in name:
            non_pretrain_ln_params.append(param)
        else:
            non_pretrain_params.append(param)
optimizer = torch.optim.AdamW([{'params': non_ln_params, 'lr': args.lr, 'weight_decay': weight_decay},
                               {'params': ln_params, 'lr': args.lr, 'weight_decay': 0},
                               {'params': non_pretrain_ln_params, 'lr': args.lr * non_ptm_lr_ratio, 'weight_decay': 0},
                               {'params': non_pretrain_params, 'lr': args.lr * non_ptm_lr_ratio,
                                'weight_decay': weight_decay}])

# callbacks
callbacks = []
callbacks.append(FitlogCallback())
callbacks.append(TorchGradClipCallback(clip_value=1))
callbacks.append(TorchWarmupCallback(warmup=args.warmup, schedule=schedule))

evaluate_dls = {}
if 'dev' in dls:
    evaluate_dls['dev'] = dls['dev']
evaluate_dls['test'] = dls['test']
allow_nested = False
if args.dataset_name.startswith('ace') or 'genia' in args.dataset_name:
    allow_nested = True

metrics = {'fast_f': FastNERMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, allow_nested=allow_nested,
                                   symmetric=symmetric)}


def evaluate_every(trainer):
    if (trainer.cur_epoch_idx >= 1 and trainer.global_forward_batches % trainer.num_batches_per_epoch == 0) or \
            trainer.global_forward_batches == trainer.n_batches:
        return True


trainer = Trainer(model=model,
                  driver='torch',
                  train_dataloader=dls.get('train'),
                  evaluate_dataloaders=evaluate_dls,
                  optimizers=optimizer,
                  callbacks=callbacks,
                  overfit_batches=0,
                  device=0,
                  n_epochs=args.n_epochs,
                  metrics=metrics,
                  monitor='f#fast_f#dev' if 'dev' in evaluate_dls else 'f#fast_f#test',
                  evaluate_every=evaluate_every,
                  evaluate_use_dist_sampler=True,
                  accumulation_steps=args.accumulation_steps,
                  fp16=True,
                  progress_bar='rich',
                  torch_kwargs={'ddp_kwargs': {'find_unused_parameters': True}, 'non_blocking': True},
                  fairscale_kwargs={'fs_type': 'sdp', 'sdp_kwargs': {'auto_refresh_trainable': False}},
                  train_fn='forward_ner', evaluate_fn='forward_ner')

trainer.run(num_train_batch_per_epoch=-1, num_eval_batch_per_dl=-1, num_eval_sanity_batch=1)

evaluator = Evaluator(model=model, dataloaders=evaluate_dls,
                      metrics={'f': NERMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, allow_nested=allow_nested,
                                              symmetric=symmetric)},
                      device=0, evaluate_fn='forward_ner')
results = evaluator.run()
fitlog.add_best_metric(results)
fitlog.finish()  # finish the logging

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train_v1.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train_re.py --n_epochs 200 --lr 3e-5 --dataset_name tplinker_nyt --accumulation_steps 4
