import torch
from fastNLP import Metric

from model.utils import _compute_f_rec_pre, decode, symmetric_decode


class NERMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, allow_nested=False, symmetric=False):
        super(NERMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        self.register_element('s_tp', 0, aggregate_method='sum')
        self.register_element('s_pre', 0, aggregate_method='sum')
        self.register_element('s_rec', 0, aggregate_method='sum')
        assert len(matrix_segs) == 1, "Only support pure entities."
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres
        self.symmetric = symmetric  # 如果为True会考虑下半区的分数

    def update(self, ent_target, scores, word_len):
        ent_scores = scores.sigmoid()  # bsz x max_len x max_len x num_class
        if self.symmetric is True:
            ent_scores = (ent_scores + ent_scores.transpose(1, 2)) / 2
        elif self.symmetric == 'min':
            ent_scores = torch.minimum(ent_scores, ent_scores.transpose(1, 2))
        span_pred = ent_scores.max(dim=-1)[0]
        if self.symmetric == 2:
            span_ents = symmetric_decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        else:
            span_ents = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        ent_preds = ent_scores.argmax(dim=-1)
        for ents, span_ent, ent_pred in zip(ent_target, span_ents, ent_preds):
            pred_spans = [(s, e) for s, e, _ in span_ent]
            spans = [(s, e) for s, e, _ in ents]
            self.s_tp += len(set(map(tuple, pred_spans)).intersection(spans))
            self.s_pre += len(pred_spans)
            self.s_rec += len(spans)

            pred_ent = set()
            for s, e, l in span_ent:
                ent_type = ent_pred[s, e]
                # ent_type = score.argmax()
                pred_ent.add((s, e, ent_type.item()))
            self.tp += len(set(map(tuple, ents)).intersection(pred_ent))
            self.pre += len(pred_ent)
            self.rec += len(ents)

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        f, rec, pre = _compute_f_rec_pre(self.s_tp, self.s_rec, self.s_pre)
        res.update({'s_f': f, 's_rec': rec, 's_pre': pre})
        return res


class FastNERMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, allow_nested=False, symmetric=False):
        super(FastNERMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')
        assert len(matrix_segs) == 1, "Only support pure entities."
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres
        self.symmetric = symmetric  # 如果为True会考虑下半区的分数

    def update(self, ent_target, scores, word_len, matrix):
        ent_scores = scores.sigmoid()  # bsz x max_len x max_len x num_class
        mask = ((ent_scores > self.ent_thres).sum(dim=-1) == 0)  # 没有任何一个超过的位置
        pred = ent_scores.argmax(dim=-1).masked_fill(mask, 0)
        pad_mask = matrix.eq(-100).sum(dim=-1) > 0
        self.pre += mask.eq(0).masked_fill(pad_mask, 0).sum()
        self.rec += (matrix == 1).sum()
        one_mask = torch.logical_or(pad_mask, matrix.sum(dim=-1) != 1)
        one_mask = torch.logical_or(one_mask, mask)
        self.tp += (pred == matrix.argmax(dim=-1)).masked_fill(one_mask, 0).sum()

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        return res
