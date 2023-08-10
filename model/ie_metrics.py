from collections import Counter

from fastNLP import Metric

from model.utils import decode, _compute_f_rec_pre, symmetric_decode


class IEMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, rel_thres, tri_thres, role_thres, arg_thres, allow_nested=False,
                 constrain=None, symmetric=False):
        # constrain: {key: set()}  其中key为event_type的index，value是这个event_type运行的role有哪些
        super(IEMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        self.register_element('r_tp', 0, aggregate_method='sum')
        self.register_element('r_pre', 0, aggregate_method='sum')
        self.register_element('r_rec', 0, aggregate_method='sum')

        self.register_element('e_tp', 0, aggregate_method='sum')
        self.register_element('e_pre', 0, aggregate_method='sum')
        self.register_element('e_rec', 0, aggregate_method='sum')

        self.register_element('rel_tp', 0, aggregate_method='sum')
        self.register_element('rel_pre', 0, aggregate_method='sum')
        self.register_element('rel_rec', 0, aggregate_method='sum')

        self.register_element('er_tp', 0, aggregate_method='sum')
        self.register_element('er_pre', 0, aggregate_method='sum')
        self.register_element('er_rec', 0, aggregate_method='sum')

        self.num_ent_label = matrix_segs['ent']
        self.num_rel_label = matrix_segs['rel']
        self.num_tri_label = matrix_segs['tri']
        self.num_role_label = matrix_segs['role']
        self.ent_thres = ent_thres
        self.rel_thres = rel_thres
        self.role_thres = role_thres
        self.tri_thres = tri_thres
        self.arg_thres = arg_thres
        self.allow_nested = allow_nested
        self.count = 0
        self.constrain = constrain
        self.symmetric = symmetric

    def update(self, tri_target, arg_target, ent_target, rel_target, scores, word_len, tokens):
        assert self.num_tri_label + self.num_role_label + self.num_ent_label + self.num_rel_label + 1 == scores.size(-1)
        if self.symmetric is True:
            scores = (scores + scores.transpose(1, 2)) / 2

        ent_logits = scores[..., :self.num_ent_label]
        rel_logits = scores[..., self.num_ent_label:self.num_ent_label + self.num_rel_label]
        arg_logits = scores[..., self.num_ent_label + self.num_rel_label]  # bsz x max_len x max_len
        tri_logits = scores[...,
                     self.num_ent_label + self.num_rel_label + 1:self.num_ent_label + self.num_rel_label + self.num_tri_label + 1]
        role_logits = scores[..., self.num_ent_label + self.num_rel_label + self.num_tri_label + 1:].cpu().numpy()
        tri_pred = tri_logits.max(dim=-1)[0]

        span_pred = ent_logits.max(dim=-1)[0]
        # [set([(s, e, r), ()]), set()]
        if self.symmetric == 2:
            tri_preds = symmetric_decode(tri_pred, word_len, allow_nested=self.allow_nested, thres=self.tri_thres)
            span_pred = symmetric_decode(span_pred, word_len, allow_nested=False, thres=self.ent_thres)
        else:
            tri_preds = decode(tri_pred, word_len, allow_nested=self.allow_nested, thres=self.tri_thres)
            span_pred = decode(span_pred, word_len, allow_nested=False, thres=self.ent_thres)

        arg_preds = decode(arg_logits, word_len, allow_nested=False, thres=self.arg_thres)
        self.count += 1
        for ins_idx, (tris, args, tri_pred, tri_score, role_score, ents, rels, spans, ent_pred, rel_pred) in \
                enumerate(zip(tri_target, arg_target, tri_preds, tri_logits.cpu().numpy(), role_logits,
                              ent_target, rel_target, span_pred, ent_logits.cpu().numpy(), rel_logits.cpu().numpy())):
            pred_ent = set()
            pred_dict = {}
            for s, e, _ in spans:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                pred_ent.add((s, e, ent_type))
                pred_dict[(s, e)] = ent_type

            tp = len(set(map(tuple, ents)).intersection(pred_ent))
            self.e_tp += tp
            self.e_pre += len(pred_ent)
            self.e_rec += len(ents)
            pred_rel = set()
            pred_ent = list(sorted(pred_ent, key=lambda x: x[0]))
            for i in range(len(pred_ent)):
                for j in range(len(pred_ent)):
                    if i != j:
                        s1, e1, _ = pred_ent[i]
                        s2, e2, _ = pred_ent[j]
                        hh_score = rel_pred[s1, s2]  # num_rel
                        tt_score = rel_pred[e1, e2]
                        score = (hh_score + tt_score) / 2
                        if score.max() < self.rel_thres:
                            continue
                        idx = score.argmax()
                        pred_rel.add(frozenset(((s1, e1), (s2, e2), idx)))
            rels = set([frozenset(((s1, e1), (s2, e2), idx)) for (s1, e1, s2, e2, idx) in rels])
            r_tp = 0
            for t1 in pred_rel:
                if t1 in rels:
                    r_tp += 1
            self.rel_tp += r_tp
            self.rel_pre += len(pred_rel)
            self.rel_rec += len(rels)

            pred_tris = set()
            pred_dict = {}
            for s, e, _ in tri_pred:
                score = tri_score[s, e]
                tri_type = score.argmax()
                pred_tris.add((s, e, tri_type))
                pred_dict[(s, e)] = tri_type
            tp = len(set(map(tuple, tris)).intersection(pred_tris))
            self.tp += tp
            self.pre += len(pred_tris)
            self.rec += len(tris)

            pred_tris = list(pred_tris)
            # 这里如果，arg_pred没有和任何trigger的分数大于某个值，那说明这个arg是不需要的
            arg_pred = [(s, e) for s, e, t in arg_preds[ins_idx]]
            pred_args = Counter()
            for i in range(len(pred_tris)):
                for j in range(len(arg_pred)):
                    s, e, t = pred_tris[i]
                    a_s, a_e = arg_pred[j]
                    hh_score = role_score[s, a_s]  # num_rel
                    tt_score = role_score[e, a_e]
                    score = (hh_score + tt_score) / 2
                    if score.max() < self.role_thres:
                        continue
                    idx = score.argmax()
                    if idx in self.constrain[t]:
                        pred_args[(a_s, a_e, idx, t)] += 1
            r_tp = 0
            args = Counter(map(tuple, args))
            for t, v in args.items():
                r_tp += int(t in pred_args)
            self.r_tp += r_tp
            self.r_pre += len(pred_args)
            self.r_rec += len(args)

            pred_args = Counter()
            for i in range(len(pred_tris)):
                for j in range(len(pred_ent)):
                    s, e, t = pred_tris[i]
                    a_s, a_e, _ = pred_ent[j]
                    hh_score = role_score[s, a_s]  # num_rel
                    tt_score = role_score[e, a_e]
                    score = (hh_score + tt_score) / 2
                    if score.max() < self.role_thres:
                        continue
                    idx = score.argmax()
                    if idx in self.constrain[t]:
                        pred_args[(a_s, a_e, idx, t)] += 1
            r_tp = 0
            args = Counter(map(tuple, args))
            for t, v in args.items():
                r_tp += int(t in pred_args)
            self.er_tp += r_tp
            self.er_pre += len(pred_args)
            self.er_rec += len(args)

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        f, rec, pre = _compute_f_rec_pre(self.r_tp, self.r_rec, self.r_pre)
        res.update({'r_f': f, 'r_rec': rec, 'r_pre': pre})

        f, rec, pre = _compute_f_rec_pre(self.e_tp, self.e_rec, self.e_pre)
        res.update({'e_f': f, 'e_rec': rec, 'e_pre': pre})

        f, rec, pre = _compute_f_rec_pre(self.rel_tp, self.rel_rec, self.rel_pre)
        res.update({'rel_f': f, 'rel_rec': rec, 'rel_pre': pre})

        f, rec, pre = _compute_f_rec_pre(self.er_tp, self.er_rec, self.er_pre)
        res.update({'er_f': f, 'er_rec': rec, 'er_pre': pre})
        return res


class ProbeIEMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, rel_thres, tri_thres, role_thres, arg_thres, allow_nested=False,
                 constrain=None, symmetric=False):
        # constrain: {key: set()}  其中key为event_type的index，value是这个event_type运行的role有哪些
        super(ProbeIEMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        self.register_element('r_tp', 0, aggregate_method='sum')
        self.register_element('r_pre', 0, aggregate_method='sum')
        self.register_element('r_rec', 0, aggregate_method='sum')

        self.register_element('e_tp', 0, aggregate_method='sum')
        self.register_element('e_pre', 0, aggregate_method='sum')
        self.register_element('e_rec', 0, aggregate_method='sum')

        self.register_element('rel_tp', 0, aggregate_method='sum')
        self.register_element('rel_pre', 0, aggregate_method='sum')
        self.register_element('rel_rec', 0, aggregate_method='sum')

        self.num_ent_label = matrix_segs['ent']
        self.num_rel_label = matrix_segs['rel']
        self.num_tri_label = matrix_segs['tri']
        self.num_role_label = matrix_segs['role']
        self.ent_thres = ent_thres
        self.rel_thres = rel_thres
        self.role_thres = role_thres
        self.tri_thres = tri_thres
        self.arg_thres = arg_thres
        self.allow_nested = allow_nested
        self.register_element('count', 0)
        self.constrain = constrain
        self.symmetric = symmetric
        self.register_element('no_found_inpred_span', 0)
        self.register_element('no_found_but_should', 0)
        self.register_element('in_ent_role', 0)

    def update(self, tri_target, arg_target, ent_target, rel_target, scores, word_len, tokens):
        assert self.num_tri_label + self.num_role_label + self.num_ent_label + self.num_rel_label + 1 == scores.size(-1)
        if self.symmetric is True:
            scores = (scores + scores.transpose(1, 2)) / 2

        ent_logits = scores[..., :self.num_ent_label]
        rel_logits = scores[..., self.num_ent_label:self.num_ent_label + self.num_rel_label]
        arg_logits = scores[..., self.num_ent_label + self.num_rel_label]  # bsz x max_len x max_len
        tri_logits = scores[...,
                     self.num_ent_label + self.num_rel_label + 1:self.num_ent_label + self.num_rel_label + self.num_tri_label + 1]
        role_logits = scores[..., self.num_ent_label + self.num_rel_label + self.num_tri_label + 1:].cpu().numpy()
        tri_pred = tri_logits.max(dim=-1)[0]

        span_pred = ent_logits.max(dim=-1)[0]
        # [set([(s, e, r), ()]), set()]
        if self.symmetric == 2:
            tri_preds = symmetric_decode(tri_pred, word_len, allow_nested=self.allow_nested, thres=self.tri_thres)
            span_pred = symmetric_decode(span_pred, word_len, allow_nested=False, thres=self.ent_thres)
        else:
            tri_preds = decode(tri_pred, word_len, allow_nested=self.allow_nested, thres=self.tri_thres)
            span_pred = decode(span_pred, word_len, allow_nested=False, thres=self.ent_thres)

        arg_preds = arg_logits >= self.arg_thres
        self.count += 1
        for ins_idx, (tris, args, tri_pred, tri_score, role_score, ents, rels, spans, ent_pred, rel_pred) in \
                enumerate(zip(tri_target, arg_target, tri_preds, tri_logits.cpu().numpy(), role_logits,
                              ent_target, rel_target, span_pred, ent_logits.cpu().numpy(), rel_logits.cpu().numpy())):
            pred_ent = set()
            pred_dict = {}
            for s, e, _ in spans:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                pred_ent.add((s, e, ent_type))
                pred_dict[(s, e)] = ent_type

            tp = len(set(map(tuple, ents)).intersection(pred_ent))
            self.e_tp += tp
            self.e_pre += len(pred_ent)
            self.e_rec += len(ents)
            pred_rel = set()
            pred_ent = list(sorted(pred_ent, key=lambda x: x[0]))
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
                            if (s2, e2, s1, e1, idx) not in pred_rel:
                                pred_rel.add((s1, e1, s2, e2, idx))
            rels = set(map(tuple, rels))
            r_tp = 0
            for t1 in pred_rel:
                if t1 in rels:
                    r_tp += 1
                elif (t1[2], t1[3], t1[0], t1[1], t1[4]) in rels:
                    r_tp += 1
            self.rel_tp += r_tp
            self.rel_pre += len(pred_rel)
            self.rel_rec += len(rels)

            pred_tris = set()
            pred_dict = {}
            for s, e, _ in tri_pred:
                score = tri_score[s, e]
                tri_type = score.argmax()
                pred_tris.add((s, e, tri_type))
                pred_dict[(s, e)] = tri_type
            tp = len(set(map(tuple, tris)).intersection(pred_tris))
            self.tp += tp
            self.pre += len(pred_tris)
            self.rec += len(tris)

            pred_tris = list(pred_tris)
            arg_pred = arg_preds[ins_idx, :word_len[ins_idx],
                       :word_len[ins_idx]].triu().nonzero().tolist()  # [(s, e), (s, e)]
            # 这里如果，arg_pred没有和任何trigger的分数大于某个值，那说明这个arg是不需要的
            pred_args = Counter()
            for i in range(len(pred_tris)):
                for j in range(len(pred_ent)):
                    s, e, t = pred_tris[i]
                    a_s, a_e, _ = pred_ent[j]
                    hh_score = role_score[s, a_s]  # num_rel
                    tt_score = role_score[e, a_e]
                    score = (hh_score + tt_score) / 2
                    pred_roles = score >= self.role_thres
                    idxes = pred_roles.nonzero()[0]
                    for idx in idxes:
                        if idx in self.constrain[t]:
                            pred_args[(a_s, a_e, idx, t)] += 1
            r_tp = 0
            args = Counter(map(tuple, args))
            for t, v in args.items():
                r_tp += int(t in pred_args)
            self.r_tp += r_tp
            self.r_pre += len(pred_args)
            self.r_rec += len(args)

            ent_spans = set([(s, e) for (s, e, t) in pred_ent])
            role_spans = set([(s, e) for (s, e) in arg_pred])
            t_role_spans = set([(s, e) for (s, e, _, _) in args])
            for span in role_spans:
                if span not in ent_spans:
                    self.no_found_inpred_span += 1
                    if span in t_role_spans:
                        self.no_found_but_should += 1
            for span in t_role_spans:
                if span not in role_spans and span in ent_spans:
                    self.in_ent_role += 1

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        f, rec, pre = _compute_f_rec_pre(self.r_tp, self.r_rec, self.r_pre)
        res.update({'r_f': f, 'r_rec': rec, 'r_pre': pre})

        f, rec, pre = _compute_f_rec_pre(self.e_tp, self.e_rec, self.e_pre)
        res.update({'e_f': f, 'e_rec': rec, 'e_pre': pre})

        f, rec, pre = _compute_f_rec_pre(self.rel_tp, self.rel_rec, self.rel_pre)
        res.update({'rel_f': f, 'rel_rec': rec, 'rel_pre': pre})

        return res
