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
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP.core.dataloaders.utils import OverfitDataLoader
import fitlog
import torch

fitlog.debug()
from data.ie_pipe import OneIEPipe
from data.padder import Torch3DMatrixPadder
from model.args import ARGS
from model.batch_sampler import ConstantTokenBatchSampler
from model.ie_metrics import ProbeIEMetric
from model.unify_model import UnifyModel

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=3e-5, type=float)
parser.add_argument('-b', '--batch_size', default=48, type=int)
parser.add_argument('-n', '--n_epochs', default=1000, type=int)
parser.add_argument('-a', '--accumulation_steps', default=1, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='ere', type=str)
parser.add_argument('--model_name', default=None, type=str)
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
    if args.dataset_name == 'cn_ace05':
        args.model_name = 'bert-base-multilingual-cased'
    else:
        args.model_name = 'bert-large-cased'

model_name = args.model_name

if 'base' in args.model_name:
    batch_size_when_max_len = 4
elif torch.cuda.mem_get_info()[-1] > 30_043_904_512 and 'xlarge' in model_name:
    batch_size_when_max_len = 12
    max_token = 2048
elif torch.cuda.mem_get_info()[-1] > 30_043_904_512:
    batch_size_when_max_len = 12
    max_token = 2048
elif 'large' in model_name and dataset_name == 'ere':
    batch_size_when_max_len = min(args.batch_size, 12)
    max_token = 1840
elif 'large' in model_name and dataset_name == 'ace05E+':
    batch_size_when_max_len = min(args.batch_size, 15)
elif 'large' in model_name and dataset_name == 'ace05E':
    batch_size_when_max_len = min(args.batch_size, 12)
    max_token = 1840
elif dataset_name == 'cn_ace05':
    batch_size_when_max_len = min(args.batch_size, 10)
    max_token = 1536

######hyper
non_ptm_lr_ratio = 10
use_at_loss = True
attn_dropout = 0.15
use_tri_bias = True
schedule = 'linear'
symmetric = True  # 没有这个值为 False，在得到trigger和role的时候是否使用对称的
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
    ent_thres = 0.5
    rel_thres = 1
else:
    arg_thres = 0.5
    tri_thres = 0.5
    role_thres = 0.5
    ent_thres = 0.5
    rel_thres = 0.5

# 如果是debug模式，就不同步到fitlog
fitlog.debug()

# 这个似乎必须是True，否则收敛起来太慢了。
if args.dataset_name == 'cn_ace05':
    eval_batch_size = 10
elif 'large' in model_name and torch.cuda.mem_get_info()[-1] < 30_043_904_512:
    eval_batch_size = 20
else:
    eval_batch_size = min(args.batch_size * 2, 32)

if 'SEARCH_ID' in os.environ:
    fitlog.set_log_dir('debug_ie_logs/')
else:
    fitlog.set_log_dir('debug_ie_logs/')
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


@cache_results('caches/ie_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name):
    if dataset_name == 'ace05E':
        pipe = OneIEPipe(model_name)
        paths = '../dataset/ace05E'
    elif dataset_name == 'ace05E+':
        pipe = OneIEPipe(model_name)
        paths = '../dataset/ace05E+'
    elif dataset_name == 'ere':
        pipe = OneIEPipe(model_name)
        paths = '../dataset/ERE'
    elif dataset_name == 'cn_ace05':
        pipe = OneIEPipe(model_name)
        paths = '../dataset/ace2005_cn_oneie'
    dl = pipe.process_from_file(paths)
    return dl, pipe.matrix_segs


dl, matrix_segs = get_data(dataset_name, model_name)

if dataset_name == 'ace05E+':
    lanuch_time = '2022-08-20-22_49_17_177633'
    args.cross_dim = 200
    args.biaffine_size = 300
    args.cross_depth = 3
    args.drop_s1_p = 0
    args.use_ln = 2
    args.use_s2 = 1
if dataset_name == 'ere':
    lanuch_time = '2022-08-20-12_31_46_380911'
    args.cross_dim = 200
    args.biaffine_size = 300
    args.cross_depth = 3
    args.drop_s1_p = 0
    args.use_ln = 2
    args.use_s2 = 1


def densify(x):
    x = x.todense()
    x = x.astype(np.float32)
    if use_at_loss is True:  # 根据是否使用 at loss 补充一个维度
        na = (x[..., matrix_segs['ent']:matrix_segs['ent'] + matrix_segs['rel']].sum(axis=-1,
                                                                                     keepdims=True) == 0).astype(
            np.float32)
        x = np.concatenate([x[..., :matrix_segs['ent']], na, x[..., matrix_segs['ent']:]], axis=-1)
        na = (x[..., -matrix_segs['role']:].sum(axis=-1, keepdims=True) == 0).astype(np.float32)
        x = np.concatenate([x[..., :-matrix_segs['role']], na, x[..., -matrix_segs['role']:]], axis=-1)
    return x


dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')
dl.apply_field(lambda x: x.todense().astype(bool), field_name='rel_mask', new_field_name='rel_mask',
               progress_bar='Densify')

print(dl)

tri_vocab = getattr(dl, 'tri_vocab')
role_vocab = getattr(dl, 'role_vocab')
ent_vocab = getattr(dl, 'ent_vocab')
rel_vocab = getattr(dl, 'rel_vocab')
print(f"{len(ent_vocab)} ent: {ent_vocab}, {len(rel_vocab)} rel: {rel_vocab} "
      f"{len(tri_vocab)} tri: {tri_vocab}, {len(role_vocab)} role: {role_vocab}, matrix_segs:{matrix_segs}")
dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=sum(matrix_segs.values()) + 2 if use_at_loss else sum(
                                                        matrix_segs.values()),
                                                    batch_size=max(32, args.batch_size)))

    if name == 'train':
        if 'large' in model_name or torch.cuda.mem_get_info()[-1] < 30_043_904_512:  # 区分 A100 和 3090 的
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=4,
                                           batch_sampler=ConstantTokenBatchSampler(ds.get_field('bpe_len').content,
                                                                                   max_token=max_token,
                                                                                   max_sentence=args.batch_size,
                                                                                   batch_size_when_max_len=batch_size_when_max_len),
                                           pin_memory=True, shuffle=True)
        else:
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                           batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                              batch_size=args.batch_size,
                                                                              num_batch_per_bucket=30),
                                           pin_memory=True, shuffle=True)
    else:
        _dl = prepare_torch_dataloader(ds, batch_size=32, num_workers=0,
                                       sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
        _dl = OverfitDataLoader(_dl, overfit_batches=-1)
    dls[name] = _dl

model = UnifyModel(model_name, matrix_segs, use_at_loss=use_at_loss, cross_dim=args.cross_dim,
                   cross_depth=args.cross_depth, biaffine_size=args.biaffine_size, use_ln=args.use_ln,
                   drop_s1_p=args.drop_s1_p, use_s2=args.use_s2, empty_rel_weight=args.empty_rel_weight,
                   attn_dropout=attn_dropout, use_tri_bias=use_tri_bias)

evaluate_dls = {
    'dev': dls.get('dev'),
    'test': dls.get('test')
}
constrain = getattr(dl, 'constrain')
metrics = {'f': ProbeIEMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, rel_thres=rel_thres, arg_thres=arg_thres,
                              tri_thres=tri_thres, role_thres=role_thres, allow_nested=False, constrain=constrain,
                              symmetric=symmetric)}

evaluator = fastNLP.Evaluator(
    model=model,
    metrics=metrics,
    dataloaders=evaluate_dls,
    evaluate_fn='forward_ie',
    device=0
)

evaluator.load_model(f'save_models/{lanuch_time}/best_so_far')
evaluator.run()

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train_v1.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train_re.py --n_epochs 200 --lr 3e-5 --dataset_name tplinker_nyt --accumulation_steps 4
