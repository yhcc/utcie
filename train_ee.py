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
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP import TorchGradClipCallback
from fastNLP.core.dataloaders.utils import OverfitDataLoader
import fitlog
import torch

from data.ee_pipe import EEPipe, EEPipe_
from data.padder import Torch3DMatrixPadder
from model.args import ARGS
from model.batch_sampler import ConstantTokenBatchSampler
from model.callbacks import FitlogCallback, TorchWarmupCallback
from model.ee_metric import EEMetric, EEMetric_, NestEEMetric
from model.unify_model import UnifyModel

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=3e-5, type=float)
parser.add_argument('-b', '--batch_size', default=12, type=int)
parser.add_argument('-n', '--n_epochs', default=50, type=int)
parser.add_argument('-a', '--accumulation_steps', default=1, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='ace05E', type=str)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--attn_dropout', default=0.15, type=float)
parser.add_argument('--cross_depth', default=3, type=int)
parser.add_argument('--cross_dim', default=200, type=int)
parser.add_argument('--use_ln', default=1, type=int)
parser.add_argument('--use_s2', default=1, type=int)
parser.add_argument('--drop_s1_p', default=0.1, type=float)
parser.add_argument('--empty_rel_weight', default=0.1, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)

args = parser.parse_args()
dataset_name = args.dataset_name
max_token = 2048
if args.model_name is None:
    if dataset_name in ('ace05E', 'ace05E+'):
        args.model_name = 'microsoft/deberta-v3-large'
    elif dataset_name == 'ere':
        args.model_name = 'microsoft/deberta-v3-large'
    else:
        args.model_name = 'microsoft/deberta-v3-large'

model_name = args.model_name

if torch.cuda.mem_get_info()[-1] > 30_043_904_512 and 'xlarge' in model_name:
    batch_size_when_max_len = 12
    max_token = 2048
elif torch.cuda.mem_get_info()[-1] > 30_043_904_512:
    batch_size_when_max_len = 12
    max_token = 2048
elif 'roberta-large' in model_name and 'ere' in dataset_name:
    batch_size_when_max_len = min(args.batch_size, 10)
    max_token = 1024
elif 'large' in model_name and 'ere' in dataset_name:
    batch_size_when_max_len = min(args.batch_size, 12)
    max_token = 1840
elif 'large' in model_name and 'ace05E+' in dataset_name:
    batch_size_when_max_len = min(args.batch_size, 15)
elif 'large' in model_name and 'ace05E' in dataset_name:
    batch_size_when_max_len = min(args.batch_size, 12)
    max_token = 2048
elif 'base' in model_name and 'ere' in dataset_name:
    batch_size_when_max_len = min(args.batch_size, 24)
    max_token = 2048
elif 'base' in model_name and 'ace05E+' in dataset_name:
    batch_size_when_max_len = min(args.batch_size, 30)
elif 'base' in model_name and 'ace05E' in dataset_name:
    batch_size_when_max_len = min(args.batch_size, 24)
    max_token = 2048

######hyper
non_ptm_lr_ratio = 10
use_at_loss = True
use_tri_bias = True
schedule = 'linear'
symmetric = True  # 没有这个值为 False，在得到trigger和role的时候是否使用对称的
ignore_top_4 = 'auto'
use_set = True  # 没有的为False
######hyper
ARGS['use_pos'] = False
ARGS['use_gelu'] = True
ARGS['s1_scale_plus'] = False
ARGS['drop_p'] = 0.4  # 没有这个参数的实验是0.4
ARGS['use_residual'] = True

if use_at_loss is True:
    arg_thres = 0.5
    tri_thres = 0.5
    role_thres = 1
else:
    arg_thres = 0.5
    tri_thres = 0.5
    role_thres = 0.5

# 如果是debug模式，就不同步到fitlog
fitlog.debug()

# 这个似乎必须是True，否则收敛起来太慢了。
if 'large' in model_name and torch.cuda.mem_get_info()[-1] < 30_043_904_512:
    eval_batch_size = 20
else:
    eval_batch_size = min(args.batch_size * 2, 32)

if 'SEARCH_ID' in os.environ:
    fitlog.set_log_dir('debug_ee_logs2/')
else:
    fitlog.set_log_dir('debug_ee_logs2/')
seed = fitlog.set_rng_seed()
os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
if 'SEARCH_ID' not in os.environ and fastNLP.get_global_rank() == 0:
    fitlog.commit(__file__)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)
if 'SEARCH_ID' in os.environ:
    fitlog.add_other(os.environ['SEARCH_ID'], name='SEARCH_ID')
    fit_id = fitlog.get_fit_id(__file__)[:8]
    if fit_id is not None:
        fitlog.add_other(name='fit_id', value=fit_id)
fitlog.add_hyper(ARGS)


@cache_results('caches/ee_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name):
    if dataset_name == 'ace05E':
        pipe = EEPipe(model_name)
        paths = '../dataset/ace05E'
    elif dataset_name == 'ace05E+':
        pipe = EEPipe(model_name)
        paths = '../dataset/ace05E+'
    elif dataset_name == 'ere':
        pipe = EEPipe(model_name)
        paths = '../dataset/ERE_text2event'
    if dataset_name == 'ace05E_':  # 不考虑对称
        pipe = EEPipe_(model_name)
        paths = '../dataset/ace05E'
    elif dataset_name == 'ace05E+_':
        pipe = EEPipe_(model_name)
        paths = '../dataset/ace05E+'
    elif dataset_name == 'ere_':
        pipe = EEPipe_(model_name)
    if dataset_name == 'oace05E':  # 以o开头的是模仿oneie的做法，一个role只出现在一个trigger那里；同时nested的entity取前面那个
        pipe = EEPipe(model_name)
        paths = '../dataset/ace05E'
        dl = pipe.process_from_file(paths, True)
        return dl, pipe.matrix_segs
    elif dataset_name == 'oace05E+':
        pipe = EEPipe(model_name)
        paths = '../dataset/ace05E+'
        dl = pipe.process_from_file(paths, True)
        return dl, pipe.matrix_segs
    elif dataset_name == 'oere':
        pipe = EEPipe(model_name)
        paths = '../dataset/ERE_text2event'
        dl = pipe.process_from_file(paths, True)
        return dl, pipe.matrix_segs
    dl = pipe.process_from_file(paths)
    return dl, pipe.matrix_segs


dl, matrix_segs = get_data(dataset_name, model_name)
tri_vocab = getattr(dl, 'tri_vocab')
role_vocab = getattr(dl, 'role_vocab')
"""
保证没有nested的
"""
for name, ds in dl.iter_datasets():
    for ins in ds:
        tri_target = ins['tri_target']
        flags = np.zeros(512)
        for s, e, _ in tri_target:
            assert all(flags[s:e + 1] == 0)
            flags[s:e + 1] += 1
        arg_target = set(ins['arg_target'])
        flags = np.zeros(512)
        for s, e in set([(s, e) for s, e, _, _ in arg_target]):
            assert all(flags[s:e + 1] == 0)
            flags[s:e + 1] += 1

if ignore_top_4 == 'auto' and dataset_name in ('ace05E', 'ace05E_', 'oace05E'):
    print(dl)
    for name, ds in dl.iter_datasets():
        if name != 'train':
            ds.drop(lambda x: int(x['sent_id'].split('-')[-1]) < 4, inplace=True)


def densify(x):
    x = x.todense()
    x = x.astype(np.float32)
    if use_at_loss is True:  # 根据是否使用 at loss 补充一个维度
        na = (x[..., -matrix_segs['role']:].sum(axis=-1, keepdims=True) == 0).astype(np.float32)
        x = np.concatenate([x[..., :matrix_segs['tri'] + 1], na, x[..., -matrix_segs['role']:]], axis=-1)
    return x


dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')
dl.apply_field(lambda x: x.todense().astype(bool), field_name='rel_mask', new_field_name='rel_mask',
               progress_bar='Densify')

print(dl)

print(f"{len(tri_vocab)} tri: {tri_vocab}, {len(role_vocab)} role: {role_vocab}, matrix_segs:{matrix_segs}")
dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=sum(matrix_segs.values()) + 1 if use_at_loss else sum(
                                                        matrix_segs.values()),
                                                    batch_size=max(eval_batch_size, args.batch_size)))

    if name == 'train':
        if torch.cuda.mem_get_info()[
            -1] < 30_043_904_512 or 'deberta-v3-large' in model_name or 'xlarge' in model_name:  # 区分 A100 和 3090 的
            # if 'deberta-v3-large' in model_name or 'xlarge' in model_name:  # 区分 A100 和 3090 的
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=4,
                                           batch_sampler=ConstantTokenBatchSampler(ds.get_field('bpe_len').content,
                                                                                   max_token=max_token,
                                                                                   max_sentence=args.batch_size,
                                                                                   batch_size_when_max_len=batch_size_when_max_len),
                                           pin_memory=True, shuffle=True)
        else:
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=4,
                                           batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                              batch_size=args.batch_size,
                                                                              num_batch_per_bucket=30),
                                           pin_memory=True, shuffle=True)
    else:
        _dl = prepare_torch_dataloader(ds, batch_size=eval_batch_size, num_workers=3,
                                       sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
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
non_ptm_lr = non_ptm_lr_ratio if non_ptm_lr_ratio < 1 else args.lr * non_ptm_lr_ratio
optimizer = torch.optim.AdamW([{'params': non_ln_params, 'lr': args.lr, 'weight_decay': 1e-2},
                               {'params': ln_params, 'lr': args.lr, 'weight_decay': 0},
                               {'params': non_pretrain_ln_params, 'lr': non_ptm_lr, 'weight_decay': 0},
                               {'params': non_pretrain_params, 'lr': non_ptm_lr, 'weight_decay': 1e-2}],
                              eps=1e-6)

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
constrain = getattr(dl, 'constrain')
metrics = {'f': EEMetric(matrix_segs=matrix_segs, arg_thres=arg_thres, tri_thres=tri_thres, role_thres=role_thres,
                         allow_nested=False, constrain=constrain, symmetric=symmetric, use_set=use_set)
if not dataset_name.endswith('_') else
EEMetric_(matrix_segs=matrix_segs, arg_thres=arg_thres, tri_thres=tri_thres, role_thres=role_thres,
          allow_nested=False, constrain=constrain, symmetric=symmetric, use_set=use_set),
           'nest_f': NestEEMetric(matrix_segs=matrix_segs, arg_thres=arg_thres, tri_thres=tri_thres,
                                  role_thres=role_thres,
                                  allow_nested=True, constrain=constrain, symmetric=symmetric, use_set=use_set)}


def evaluate_every(trainer):
    if trainer.cur_epoch_idx >= 10 and trainer.global_forward_batches % trainer.num_batches_per_epoch == 0:
        return True


def monitor(results):
    return results['f#f#dev'] + results['r_f#f#dev']


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
                  evaluate_use_dist_sampler=True,
                  accumulation_steps=args.accumulation_steps,
                  fp16=True,
                  progress_bar='rich',
                  train_fn='forward_ee', evaluate_fn='forward_ee')

trainer.run(num_train_batch_per_epoch=-1, num_eval_batch_per_dl=-1, num_eval_sanity_batch=0)
fitlog.finish()  # finish the logging

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train_v1.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train_re.py --n_epochs 200 --lr 3e-5 --dataset_name tplinker_nyt --accumulation_steps 4
