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
from fastNLP import Trainer
from fastNLP import TorchGradClipCallback
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP.core.dataloaders.utils import OverfitDataLoader
import fitlog
import torch

from data.padder import Torch3DMatrixPadder
from data.sym_re_pipe import RePipe_, _RePipe, RRePipe_
from model.args import ARGS
from model.batch_sampler import ConstantTokenBatchSampler
from model.callbacks import FitlogCallback, TorchWarmupCallback
from model.re_metrics import ReMetric, OneRelMetric, RReMetric
from model.unify_model import UnifyModel

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=3e-5, type=float)
parser.add_argument('-b', '--batch_size', default=16, type=int)
parser.add_argument('-n', '--n_epochs', default=100, type=int)
parser.add_argument('-a', '--accumulation_steps', default=1, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='ace2005', type=str)
# albert-xxlarge-v1
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--attn_dropout', default=0.15, type=float)
parser.add_argument('--cross_depth', default=3, type=int)
parser.add_argument('--cross_dim', default=200, type=int)
parser.add_argument('--use_ln', default=0, type=int)
parser.add_argument('--use_s2', default=1, type=int)
parser.add_argument('--use_gelu', default=0, type=int)
parser.add_argument('--use_sym_rel', default=2, type=int)
parser.add_argument('--drop_s1_p', default=0, type=float)
parser.add_argument('--empty_rel_weight', default=0.1, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--symmetric', default=True, type=int)  # 没有这个值的记录，为False, 这个用来控制选择entity的时候是否选择 上下三角相加
parser.add_argument('--use_tri_bias', default=True, type=int)
parser.add_argument('--use_size_embed', default=False, type=int)  # 没有这个值的记录，为False, 这个用来控制选择entity的时候是否选择 上下三角相加
parser.add_argument('--hidden_dropout', default=None, type=float)  # pretrain model是否没有dropout，给albert用的

args = parser.parse_args()
dataset_name = args.dataset_name
if args.model_name is None:
    if dataset_name in ('sciere_', '_sciere', 'rsciere_'):
        args.model_name = 'allenai/scibert_scivocab_uncased'
    elif dataset_name in ('ace2005_', '_ace2005', 'race2005_'):
        args.model_name = 'bert-base-uncased'
        # model_name = 'albert-xxlarge-v1'
    elif dataset_name in ('tplinker_nyt', 'tplinker_webnlg'):
        args.model_name = 'bert-base-cased'
    elif dataset_name in ('onerel_nyt', 'onerel_webnlg', 'onerel_webnlg_'):
        args.model_name = 'bert-base-cased'
    else:
        raise RuntimeError

model_name = args.model_name
if args.symmetric:
    args.symmetric = True
else:
    args.symmetric = False
if args.use_tri_bias:
    args.use_tri_bias = True
else:
    args.use_tri_bias = False
symmetric = args.symmetric
use_tri_bias = args.use_tri_bias
######hyper
non_ptm_lr_ratio = 10
use_at_loss = True
schedule = 'linear'
weight_decay = 1e-2  # 没有这个值的记录，这个值为1e-2
use_seed = None  # 没有这个值的记录，这个值为0
add_sym = True  # 如果为True，在预测了对称关系的时候，自动加入
######hyper
ARGS['use_pos'] = False
ARGS['use_gelu'] = args.use_gelu
ARGS['s1_scale_plus'] = False
ARGS['drop_p'] = 0.4  # 没有这个参数的实验是0.4
ARGS['use_size_embed'] = args.use_size_embed
ARGS['size_embed_dim'] = 50
ARGS['hidden_dropout'] = args.hidden_dropout

if use_at_loss is True:
    rel_thres = 1
    ent_thres = 0.5
else:
    rel_thres = 0.5
    ent_thres = 0.5

# 如果是debug模式，就不同步到fitlog
fitlog.debug()

# 这个似乎必须是True，否则收敛起来太慢了。
eval_batch_size = min(args.batch_size * 2, 32)
if 'SEARCH_ID' in os.environ:
    fitlog.set_log_dir('debug_re_logs/')
    fitlog.add_other(fitlog.get_fit_id(__file__)[:8], name='fit_id')
else:
    fitlog.set_log_dir('debug_re_logs/')
seed = fitlog.set_rng_seed()
os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
if 'SEARCH_ID' not in os.environ and fastNLP.get_global_rank() == 0:
    fitlog.commit(__file__)
fitlog.add_hyper(args)
fitlog.add_hyper(ARGS)
fitlog.add_hyper_in_file(__file__)
if 'SEARCH_ID' in os.environ:
    fitlog.add_other(os.environ['SEARCH_ID'], name='SEARCH_ID')


@cache_results('caches/re_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name, use_sym=True, max_length=256):
    if dataset_name == 'sciere_':
        pipe = RePipe_(model_name, use_sym=use_sym, max_length=max_length)
        paths = '../dataset/UniRE_SciERC'
    elif dataset_name == 'ace2005_':  # 'ace2005_'就是如果左边短了就让右边多一点
        pipe = RePipe_(model_name, use_sym=use_sym, max_length=max_length)
        paths = '../dataset/UniRE_ace2005'
    if dataset_name == '_sciere':  # '_sciere'就是如果短了就短了把
        pipe = _RePipe(model_name, use_sym=use_sym, max_length=max_length)
        paths = '../dataset/UniRE_SciERC'
    elif dataset_name == '_ace2005':
        pipe = _RePipe(model_name, use_sym=use_sym, max_length=max_length)
        paths = '../dataset/UniRE_ace2005'
    elif dataset_name == 'race2005_':  # lower cased
        pipe = RRePipe_(model_name, use_sym=use_sym, max_length=max_length)
        paths = '../dataset/UniRE_ace2005'
    elif dataset_name == 'rsciere_':  # lower cased
        pipe = RRePipe_(model_name, use_sym=use_sym, max_length=max_length)
        paths = '../dataset/UniRE_SciERC'
    dl = pipe.process_from_file(paths)
    return dl, pipe.matrix_segs


dl, matrix_segs = get_data(dataset_name, model_name, use_sym=True, max_length=256)


def densify(x):
    x = x.todense()
    x = x.astype(np.float32)
    if use_at_loss is True:  # 根据是否使用 at loss 补充一个维度
        na = (x[..., matrix_segs['ent']:].sum(axis=-1, keepdims=True) == 0).astype(np.float32)
        x = np.concatenate([x[..., :matrix_segs['ent']], na, x[..., matrix_segs['ent']:]], axis=-1)

    return x


dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')
dl.apply_field(lambda x: x.todense().astype(bool), field_name='rel_mask', new_field_name='rel_mask',
               progress_bar='Densify')

print(dl)
ent_vocab = getattr(dl, 'ent_vocab')
rel_vocab = getattr(dl, 'rel_vocab')
sym_rels = set()
if add_sym:
    _sym_rels = getattr(dl, 'symmetric_rels')
    if 'ace2005' in dataset_name:
        for rel in _sym_rels:
            sym_rels.add(rel_vocab.to_index(rel))
    if 'sciere' in dataset_name:
        for rel in _sym_rels:
            sym_rels.add(rel_vocab.to_index(rel))

print(f"{len(ent_vocab)} ent: {ent_vocab}, {len(rel_vocab)} rel: {rel_vocab}, matrix_segs:{matrix_segs}")
dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=sum(matrix_segs.values()) + 1 if use_at_loss else sum(
                                                        matrix_segs.values()),
                                                    batch_size=max(eval_batch_size, args.batch_size)))

    if name == 'train':
        if 'albert' in model_name:
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                           batch_sampler=ConstantTokenBatchSampler(ds.get_field('bpe_len').content,
                                                                                   max_token=4096,
                                                                                   max_sentence=args.batch_size,
                                                                                   batch_size_when_max_len=min(
                                                                                       14 if dataset_name.startswith(
                                                                                           '_') else 16,
                                                                                       args.batch_size)),
                                           pin_memory=True, shuffle=True)
        elif args.accumulation_steps == 2:  # 专门给ace2005测试大batch的
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                           pin_memory=True, shuffle=True)

        else:
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                           batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                              batch_size=args.batch_size,
                                                                              num_batch_per_bucket=30,
                                                                              seed=seed if use_seed is None else 0),
                                           pin_memory=True, shuffle=True)
    else:
        _dl = prepare_torch_dataloader(ds, batch_size=eval_batch_size, num_workers=0,
                                       sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
        if 'LOCAL_RANK' not in os.environ:
            _dl = OverfitDataLoader(_dl, overfit_batches=-1)
    dls[name] = _dl

model = UnifyModel(model_name, matrix_segs, use_at_loss=use_at_loss, cross_dim=args.cross_dim,
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

evaluate_dls = {
    'dev': dls.get('dev'),
    'test': dls.get('test')
}
allow_nested = True
# V19_debug_ner中早于log_20220902_145110这个时间的加入context的实验都是使用的 monitor = 'r_f#f#dev' 以及 allow_nested=True
# 原因是由于之前写的是 if dataset_name in ['ace2004', 'ace2005', 'sciere']
if 'sciere' in dataset_name or 'ace200' in dataset_name:
    allow_nested = False


    def monitor(results):
        return results['f#f#dev'] + results['r_f#f#dev']
else:
    monitor = 'r_f#f#dev'
if dataset_name.startswith('onerel') or dataset_name.startswith('tplinker'):
    metrics = {'f': OneRelMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, rel_thres=rel_thres,
                                 allow_nested=allow_nested, symmetric=symmetric),
               }
    fitlog.add_other(name='info', value='new_metric')
else:
    metrics = {'f': ReMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, rel_thres=rel_thres,
                             allow_nested=allow_nested, symmetric=symmetric, sym_rels=sym_rels,
                             use_sym_rel=args.use_sym_rel)
    if not dataset_name.startswith('r') else
    RReMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, rel_thres=rel_thres,
              allow_nested=allow_nested, symmetric=symmetric, sym_rels=sym_rels, use_sym_rel=args.use_sym_rel)}


def evaluate_every(trainer):
    if trainer.cur_epoch_idx >= 10 and trainer.global_forward_batches % trainer.num_batches_per_epoch == 0:
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
                  monitor=monitor,
                  evaluate_every=evaluate_every,
                  accumulation_steps=args.accumulation_steps,
                  fp16=True,
                  progress_bar='rich',
                  train_fn='forward_re', evaluate_fn='forward_re',
                  torch_kwargs={'ddp_kwargs': {'find_unused_parameters': True}},
                  )

trainer.run(num_train_batch_per_epoch=-1, num_eval_batch_per_dl=-1, num_eval_sanity_batch=1)
fitlog.finish()  # finish the logging

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train_sym_re.py --cross_depth 3 --use_ln 2 --empty_rel_weight 0.1 -b 32 -n 70 --lr 3e-5 --cross_dim 200 --biaffine_size 300 --drop_s1_p 0 -d race2005_ --model_name albert-xxlarge-v1 --use_size_embed 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train_re.py --n_epochs 200 --lr 3e-5 --dataset_name tplinker_nyt --accumulation_steps 4
