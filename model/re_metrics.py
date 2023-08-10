from fastNLP import Metric
import numpy as np

from model.utils import decode, _compute_f_rec_pre, symmetric_decode


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class ReMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, rel_thres, allow_nested=False, symmetric=False, sym_rels=None,
                 use_sym_rel=0):
        super(ReMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        self.register_element('r_tp', 0, aggregate_method='sum')
        self.register_element('r_pre', 0, aggregate_method='sum')
        self.register_element('r_rec', 0, aggregate_method='sum')

        self.num_ent_label = matrix_segs['ent']
        self.num_rel_label = matrix_segs['rel']
        self.ent_thres = ent_thres
        self.rel_thres = rel_thres
        self.allow_nested = allow_nested
        self.count = 0
        self.symmetric = symmetric
        if sym_rels is None:
            sym_rels = set()
        self.sym_rels = sym_rels
        self.use_sym_rel = use_sym_rel  # 如果为1的话，对于对称关系，必须前后都大于阈值才加入
        print(f"Using sysmetric relations {sym_rels}")

    def update(self, ent_target, rel_target, scores, word_len, matrix):
        # for i, _ent_target in enumerate(ent_target):
        #     for s, e, t in _ent_target:
        #         assert matrix[i, s, e, t] == 1
        # for i, _rel_target in enumerate(rel_target):
        #     for s1, e1, t1, s2, e2, t2, t in _rel_target:
        #         assert matrix[i, s1, s2, t+self.num_ent_label+1] == 1
        #         assert matrix[i, e1, e2, t+self.num_ent_label+1] == 1
        self.count += 1
        assert self.num_ent_label + self.num_rel_label == scores.size(-1)
        rel_logits = scores[..., self.num_ent_label:].cpu().numpy()
        # scores = scores.sigmoid()
        if self.ent_thres != 1:
            ent_scores = scores[..., :self.num_ent_label].sigmoid()
        else:
            ent_scores = scores[..., :self.num_ent_label]
        if self.symmetric is True:
            ent_scores = (ent_scores + ent_scores.transpose(1, 2)) / 2
        rel_scores = scores[..., self.num_ent_label:]
        # if self.count > 100:
        #     import pdb
        #     pdb.set_trace()
        span_pred = ent_scores.max(dim=-1)[0]
        if self.symmetric == 2:
            span_pred = symmetric_decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        else:
            span_pred = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        # if self.count >= 200:
        #     import pdb
        #     pdb.set_trace()
        ins_idx = 0
        for ents, rels, spans, ent_pred, rel_pred in zip(ent_target, rel_target, span_pred,
                                                         ent_scores.cpu().numpy(), rel_scores.cpu().numpy()):
            pred_ent = set()
            pred_dict = {}
            for s, e, _ in spans:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                pred_ent.add((s, e, ent_type))
                pred_dict[(s, e)] = ent_type

            tp = len(set(map(tuple, ents)).intersection(pred_ent))
            self.tp += tp
            self.pre += len(pred_ent)
            self.rec += len(ents)
            pred_rel = set()
            pred_ent = list(pred_ent)
            for i in range(len(pred_ent)):
                for j in range(len(pred_ent)):
                    if i != j:
                        s1, e1, _ = pred_ent[i]
                        s2, e2, _ = pred_ent[j]
                        hh_score = rel_pred[s1, s2]  # num_rel
                        tt_score = rel_pred[e1, e2]
                        score = (hh_score + tt_score) / 2
                        pred_rels = score >= self.rel_thres
                        idxes = pred_rels.nonzero()[0]
                        for idx in idxes:
                            if self.use_sym_rel == 0:
                                pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))
                                if idx in self.sym_rels:
                                    pred_rel.add((s2, e2, pred_dict[(s2, e2)], s1, e1, pred_dict[(s1, e1)], idx))
                            elif self.use_sym_rel == 1:
                                if idx in self.sym_rels:
                                    if rel_pred[s2, s1, idx] + rel_pred[e2, e1, idx] >= self.rel_thres * 2:
                                        pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))
                                        pred_rel.add((s2, e2, pred_dict[(s2, e2)], s1, e1, pred_dict[(s1, e1)], idx))
                                else:
                                    pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))
                            else:
                                if idx in self.sym_rels:
                                    pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))
                                    pred_rel.add((s2, e2, pred_dict[(s2, e2)], s1, e1, pred_dict[(s1, e1)], idx))
                                else:
                                    pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))

            r_tp = len(set(map(tuple, rels)).intersection(pred_rel))
            # if self.count>=200 and len(pred_rel) != len(rels):
            #     import pdb
            #     pdb.set_trace()
            self.r_tp += r_tp
            self.r_pre += len(pred_rel)
            self.r_rec += len(rels)
            ins_idx += 1

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        f, rec, pre = _compute_f_rec_pre(self.r_tp, self.r_rec, self.r_pre)
        res.update({'r_f': f, 'r_rec': rec, 'r_pre': pre})
        return res


class OneRelMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, rel_thres, allow_nested=False, symmetric=False):
        super(OneRelMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        self.register_element('r_tp', 0, aggregate_method='sum')
        self.register_element('r_pre', 0, aggregate_method='sum')
        self.register_element('r_rec', 0, aggregate_method='sum')

        self.num_ent_label = matrix_segs['ent']
        self.num_rel_label = matrix_segs['rel']
        self.ent_thres = ent_thres
        self.rel_thres = rel_thres
        self.allow_nested = allow_nested
        self.count = 0
        self.symmetric = symmetric

    def update(self, ent_target, rel_target, scores, word_len, raw_words):
        # for i, _ent_target in enumerate(ent_target):
        #     for s, e, t in _ent_target:
        #         assert matrix[i, s, e, t] == 1
        # for i, _rel_target in enumerate(rel_target):
        #     for s1, e1, t1, s2, e2, t2, t in _rel_target:
        #         assert matrix[i, s1, s2, t+self.num_ent_label+1] == 1
        #         assert matrix[i, e1, e2, t+self.num_ent_label+1] == 1
        self.count += 1
        assert self.num_ent_label + self.num_rel_label == scores.size(-1)
        rel_logits = scores[..., self.num_ent_label:].cpu().numpy()
        # scores = scores.sigmoid()
        if self.ent_thres != 1:
            ent_scores = scores[..., :self.num_ent_label].sigmoid()
        else:
            ent_scores = scores[..., :self.num_ent_label]
        if self.symmetric:
            ent_scores = (ent_scores + ent_scores.transpose(1, 2)) / 2
        rel_scores = scores[..., self.num_ent_label:]
        # if self.count > 100:
        #     import pdb
        #     pdb.set_trace()
        span_pred = ent_scores.max(dim=-1)[0]
        span_pred = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        # if self.count >= 200:
        #     import pdb
        #     pdb.set_trace()
        ins_idx = 0
        for ents, rels, spans, ent_pred, rel_pred, _raw_words in zip(ent_target, rel_target, span_pred,
                                                                     ent_scores.cpu().numpy(), rel_scores.cpu().numpy(),
                                                                     raw_words):
            pred_ent = set()
            pred_dict = {}
            for s, e, _ in spans:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                if score[ent_type] >= self.ent_thres:
                    pred_ent.add((s, e, ent_type))
                    pred_dict[(s, e)] = ent_type

            tp = len(set(map(tuple, ents)).intersection(pred_ent))
            self.tp += tp
            self.pre += len(pred_ent)
            self.rec += len(ents)
            pred_rel = set()
            pred_ent = list(pred_ent)
            for i in range(len(pred_ent)):
                for j in range(len(pred_ent)):
                    if i != j:
                        s1, e1, _ = pred_ent[i]
                        s2, e2, _ = pred_ent[j]
                        hh_score = rel_pred[s1, s2]  # num_rel
                        tt_score = rel_pred[e1, e2]
                        score = (hh_score + tt_score) / 2
                        pred_rels = score >= self.rel_thres
                        idxes = pred_rels.nonzero()[0]
                        for idx in idxes:
                            if ' '.join(_raw_words[s1:e1 + 1]) != ' '.join(_raw_words[s2:e2 + 1]):
                                pred_rel.add((' '.join(_raw_words[s1:e1 + 1]), idx, ' '.join(_raw_words[s2:e2 + 1])))
            r_tp = len(set(map(tuple, rels)).intersection(pred_rel))
            # if self.count>=200 and len(pred_rel) != len(rels):
            #     import pdb
            #     pdb.set_trace()
            self.r_tp += r_tp
            self.r_pre += len(pred_rel)
            self.r_rec += len(rels)
            ins_idx += 1

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        f, rec, pre = _compute_f_rec_pre(self.r_tp, self.r_rec, self.r_pre)
        res.update({'r_f': f, 'r_rec': rec, 'r_pre': pre})
        return res


class RReMetric(Metric):
    # 会考虑有reverse关系的
    def __init__(self, matrix_segs, ent_thres, rel_thres, allow_nested=False, symmetric=False, sym_rels=None,
                 use_sym_rel=0):
        super(RReMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        self.register_element('r_tp', 0, aggregate_method='sum')
        self.register_element('r_pre', 0, aggregate_method='sum')
        self.register_element('r_rec', 0, aggregate_method='sum')
        self.matrix_segs = matrix_segs

        self.num_ent_label = matrix_segs['ent']
        self.num_rel_label = matrix_segs['rel']
        self.num_sym_rel = matrix_segs['rel'] - matrix_segs['r_rel']
        self.ent_thres = ent_thres
        self.rel_thres = rel_thres
        self.allow_nested = allow_nested
        self.count = 0
        self.symmetric = symmetric
        if sym_rels is None:
            sym_rels = set()
        self.sym_rels = sym_rels
        self.use_sym_rel = use_sym_rel  # 如果为1的话，对于对称关系，必须前后都大于阈值才加入
        print(f"Using sysmetric relations {sym_rels}")

    def update(self, ent_target, rel_target, scores, word_len, matrix):
        # for i, _ent_target in enumerate(ent_target):
        #     for s, e, t in _ent_target:
        #         assert matrix[i, s, e, t] == 1
        # for i, _rel_target in enumerate(rel_target):
        #     for s1, e1, t1, s2, e2, t2, t in _rel_target:
        #         assert matrix[i, s1, s2, t+self.num_ent_label+1] == 1
        #         assert matrix[i, e1, e2, t+self.num_ent_label+1] == 1
        self.count += 1
        assert sum(self.matrix_segs.values()) == scores.size(-1)
        # scores = scores.sigmoid()
        if self.ent_thres != 1:
            ent_scores = scores[..., :self.num_ent_label].sigmoid()
        else:
            ent_scores = scores[..., :self.num_ent_label]
        if self.symmetric is True:
            ent_scores = (ent_scores + ent_scores.transpose(1, 2)) / 2
        rel_scores = scores[..., self.num_ent_label:]
        # if self.use_sym_rel <  3:
        #     rel_scores[..., self.num_sym_rel:self.num_rel_label] = (rel_scores[..., self.num_sym_rel:self.num_rel_label] +
        #                                                             rel_scores[..., self.num_rel_label:].transpose(1, 2))/2
        #     rel_scores = rel_scores[..., :self.num_rel_label]

        # if self.count > 100:
        #     import pdb
        #     pdb.set_trace()
        span_pred = ent_scores.max(dim=-1)[0]
        if self.symmetric is True:
            span_pred = symmetric_decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        else:
            span_pred = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        # if self.count >= 200:
        #     import pdb
        #     pdb.set_trace()
        ins_idx = 0
        for ents, rels, spans, ent_pred, rel_pred in zip(ent_target, rel_target, span_pred,
                                                         ent_scores.cpu().numpy(), rel_scores.cpu().numpy()):
            pred_ent = set()
            pred_dict = {}
            for s, e, _ in spans:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                pred_ent.add((s, e, ent_type))
                pred_dict[(s, e)] = ent_type

            tp = len(set(map(tuple, ents)).intersection(pred_ent))
            self.tp += tp
            self.pre += len(pred_ent)
            self.rec += len(ents)
            pred_rel = set()
            pred_ent = list(pred_ent)
            for i in range(len(pred_ent)):
                for j in range(len(pred_ent)):
                    if i != j:
                        s1, e1, _ = pred_ent[i]
                        s2, e2, _ = pred_ent[j]
                        hh_score = rel_pred[s1, s2, :self.num_rel_label]  # num_rel
                        tt_score = rel_pred[e1, e2, :self.num_rel_label]
                        score = (hh_score + tt_score) / 2
                        pred_rels = score >= self.rel_thres
                        idxes = pred_rels.nonzero()[0]

                        hh_score = rel_pred[s2, s1, self.num_rel_label:]
                        tt_score = rel_pred[e2, e1, self.num_rel_label:]
                        score = (hh_score + tt_score) / 2
                        pred_rels = score >= self.rel_thres
                        idxes = list(idxes) + list(pred_rels.nonzero()[0] + self.num_sym_rel)

                        for idx in idxes:
                            if self.use_sym_rel == 0:
                                pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))
                                if idx in self.sym_rels:
                                    pred_rel.add((s2, e2, pred_dict[(s2, e2)], s1, e1, pred_dict[(s1, e1)], idx))
                            elif self.use_sym_rel == 1:
                                if idx in self.sym_rels:
                                    if rel_pred[s2, s1, idx] + rel_pred[e2, e1, idx] >= self.rel_thres * 2:
                                        pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))
                                        pred_rel.add((s2, e2, pred_dict[(s2, e2)], s1, e1, pred_dict[(s1, e1)], idx))
                                else:
                                    pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))
                            else:
                                if idx in self.sym_rels:
                                    pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))
                                    pred_rel.add((s2, e2, pred_dict[(s2, e2)], s1, e1, pred_dict[(s1, e1)], idx))
                                else:
                                    pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))

            r_tp = len(set(map(tuple, rels)).intersection(pred_rel))
            self.r_tp += r_tp
            self.r_pre += len(pred_rel)
            self.r_rec += len(rels)
            ins_idx += 1

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        f, rec, pre = _compute_f_rec_pre(self.r_tp, self.r_rec, self.r_pre)
        res.update({'r_f': f, 'r_rec': rec, 'r_pre': pre})
        return res


class ValidReMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, rel_thres, valid_jump, allow_nested=False):
        super(ValidReMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        self.register_element('r_tp', 0, aggregate_method='sum')
        self.register_element('r_pre', 0, aggregate_method='sum')
        self.register_element('r_rec', 0, aggregate_method='sum')

        self.num_ent_label = matrix_segs['ent']
        self.num_rel_label = matrix_segs['rel']
        self.ent_thres = ent_thres
        self.rel_thres = rel_thres
        self.allow_nested = allow_nested
        self.count = 0
        self.valid_jump = valid_jump

    def update(self, ent_target, rel_target, scores, word_len, matrix):
        # for i, _ent_target in enumerate(ent_target):
        #     for s, e, t in _ent_target:
        #         assert matrix[i, s, e, t] == 1
        # for i, _rel_target in enumerate(rel_target):
        #     for s1, e1, t1, s2, e2, t2, t in _rel_target:
        #         assert matrix[i, s1, s2, t+self.num_ent_label+1] == 1
        #         assert matrix[i, e1, e2, t+self.num_ent_label+1] == 1
        self.count += 1
        assert self.num_ent_label + self.num_rel_label == scores.size(-1)
        rel_logits = scores[..., self.num_ent_label:].cpu().numpy()
        # scores = scores.sigmoid()
        if self.ent_thres != 1:
            ent_scores = scores[..., :self.num_ent_label].sigmoid()
        else:
            ent_scores = scores[..., :self.num_ent_label]
        rel_scores = scores[..., self.num_ent_label:]
        # if self.count > 100:
        #     import pdb
        #     pdb.set_trace()
        span_pred = ent_scores.max(dim=-1)[0]
        span_pred = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        # if self.count >= 200:
        #     import pdb
        #     pdb.set_trace()
        ins_idx = 0
        for ents, rels, spans, ent_pred, rel_pred in zip(ent_target, rel_target, span_pred,
                                                         ent_scores.cpu().numpy(), rel_scores.cpu().numpy()):
            pred_ent = set()
            pred_dict = {}
            for s, e, _ in spans:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                if score[ent_type] >= self.ent_thres:
                    pred_ent.add((s, e, ent_type))
                    pred_dict[(s, e)] = ent_type

            tp = len(set(map(tuple, ents)).intersection(pred_ent))
            self.tp += tp
            self.pre += len(pred_ent)
            self.rec += len(ents)
            pred_rel = set()
            pred_ent = list(pred_ent)
            for i in range(len(pred_ent)):
                for j in range(len(pred_ent)):
                    if i != j:
                        s1, e1, _ = pred_ent[i]
                        s2, e2, _ = pred_ent[j]
                        hh_score = rel_pred[s1, s2]  # num_rel
                        tt_score = rel_pred[e1, e2]
                        score = (hh_score + tt_score) / 2
                        pred_rels = score >= self.rel_thres
                        idxes = pred_rels.nonzero()[0]
                        for idx in idxes:
                            if idx in self.valid_jump['test'].get((pred_dict[(s1, e1)], pred_dict[(s2, e2)]), {}):
                                pred_rel.add((s1, e1, pred_dict[(s1, e1)], s2, e2, pred_dict[(s2, e2)], idx))
            r_tp = len(set(map(tuple, rels)).intersection(pred_rel))
            self.r_tp += r_tp
            self.r_pre += len(pred_rel)
            self.r_rec += len(rels)
            ins_idx += 1

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        f, rec, pre = _compute_f_rec_pre(self.r_tp, self.r_rec, self.r_pre)
        res.update({'r_f': f, 'r_rec': rec, 'r_pre': pre})
        return res
