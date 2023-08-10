from collections import Counter

from fastNLP import Metric

from model.utils import decode, _compute_f_rec_pre, symmetric_decode


class EEMetric(Metric):
    def __init__(self, matrix_segs, tri_thres, role_thres, arg_thres, allow_nested=False,
                 constrain=None, symmetric=False, use_set=True):
        # constrain: {key: set()}  其中key为event_type的index，value是这个event_type运行的role有哪些
        super(EEMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        self.register_element('r_tp', 0, aggregate_method='sum')
        self.register_element('r_pre', 0, aggregate_method='sum')
        self.register_element('r_rec', 0, aggregate_method='sum')

        self.num_tri_label = matrix_segs['tri']
        self.num_role_label = matrix_segs['role']
        self.role_thres = role_thres
        self.tri_thres = tri_thres
        self.arg_thres = arg_thres
        self.allow_nested = allow_nested
        self.count = 0
        self.constrain = constrain
        self.symmetric = symmetric
        self.use_set = use_set

    def update(self, tri_target, arg_target, scores, word_len):
        assert self.num_tri_label + self.num_role_label == scores.size(-1) - 1
        if self.symmetric is True:
            scores = (scores + scores.transpose(1, 2)) / 2

        arg_logits = scores[..., 0]  # bsz x max_len x max_len
        tri_logits = scores[..., 1:self.num_tri_label + 1]
        role_logits = scores[..., self.num_tri_label + 1:].cpu().numpy()
        tri_pred = tri_logits.max(dim=-1)[0]

        # [set([(s, e, r), ()]), set()]
        if self.symmetric == 2:
            tri_preds = symmetric_decode(tri_pred, word_len, allow_nested=self.allow_nested, thres=self.tri_thres)
        else:
            tri_preds = decode(tri_pred, word_len, allow_nested=self.allow_nested, thres=self.tri_thres)
        arg_preds = decode(arg_logits, word_len, allow_nested=False, thres=self.arg_thres)

        self.count += 1
        for ins_idx, (tris, args, tri_pred, tri_score, role_score) in enumerate(zip(tri_target, arg_target, tri_preds,
                                                                                    tri_logits.cpu().numpy(),
                                                                                    role_logits)):
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
            arg_pred = [(s, e) for s, e, t in arg_preds[ins_idx]]  # [(s, e), (s, e)]
            # 这里如果，arg_pred没有和任何trigger的分数大于某个值，那说明这个arg是不需要的
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
                if self.use_set:
                    r_tp += int(t in pred_args)
                else:
                    r_tp += min(pred_args[t], v)
            self.r_tp += r_tp
            if self.use_set:
                self.r_pre += len(pred_args)
                self.r_rec += len(args)
            else:
                self.r_pre += sum(pred_args.values())
                self.r_rec += sum(args.values())

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        f, rec, pre = _compute_f_rec_pre(self.r_tp, self.r_rec, self.r_pre)
        res.update({'r_f': f, 'r_rec': rec, 'r_pre': pre})
        return res


class NestEEMetric(Metric):
    def __init__(self, matrix_segs, tri_thres, role_thres, arg_thres, allow_nested=True,
                 constrain=None, symmetric=False, use_set=True):
        # constrain: {key: set()}  其中key为event_type的index，value是这个event_type运行的role有哪些
        super(NestEEMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        self.register_element('r_tp', 0, aggregate_method='sum')
        self.register_element('r_pre', 0, aggregate_method='sum')
        self.register_element('r_rec', 0, aggregate_method='sum')

        self.num_tri_label = matrix_segs['tri']
        self.num_role_label = matrix_segs['role']
        self.role_thres = role_thres
        self.tri_thres = tri_thres
        self.arg_thres = arg_thres
        self.allow_nested = allow_nested
        self.count = 0
        self.constrain = constrain
        self.symmetric = symmetric
        self.use_set = use_set

    def update(self, tri_target, arg_target, scores, word_len):
        assert self.num_tri_label + self.num_role_label == scores.size(-1) - 1
        if self.symmetric is True:
            scores = (scores + scores.transpose(1, 2)) / 2

        arg_logits = scores[..., 0]  # bsz x max_len x max_len
        tri_logits = scores[..., 1:self.num_tri_label + 1]
        role_logits = scores[..., self.num_tri_label + 1:].cpu().numpy()
        tri_pred = tri_logits.max(dim=-1)[0]

        # [set([(s, e, r), ()]), set()]
        if self.symmetric == 2:
            tri_preds = symmetric_decode(tri_pred, word_len, allow_nested=self.allow_nested, thres=self.tri_thres)
        else:
            tri_preds = decode(tri_pred, word_len, allow_nested=self.allow_nested, thres=self.tri_thres)
        arg_preds = decode(arg_logits, word_len, allow_nested=True, thres=self.arg_thres)

        self.count += 1
        for ins_idx, (tris, args, tri_pred, tri_score, role_score) in enumerate(zip(tri_target, arg_target, tri_preds,
                                                                                    tri_logits.cpu().numpy(),
                                                                                    role_logits)):
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
            arg_pred = [(s, e) for s, e, t in arg_preds[ins_idx]]  # [(s, e), (s, e)]
            # 这里如果，arg_pred没有和任何trigger的分数大于某个值，那说明这个arg是不需要的
            pred_args = Counter()
            for i in range(len(pred_tris)):
                for j in range(len(arg_pred)):
                    s, e, t = pred_tris[i]
                    a_s, a_e = arg_pred[j]
                    hh_score = role_score[s, a_s]  # num_rel
                    tt_score = role_score[e, a_e]
                    score = (hh_score + tt_score) / 2
                    # pred_roles = score >= self.role_thres
                    if score.max() < self.role_thres:
                        continue
                    idx = score.argmax()
                    if idx in self.constrain[t]:
                        pred_args[(a_s, a_e, idx, t)] += 1
            r_tp = 0
            args = Counter(map(tuple, args))
            for t, v in args.items():
                if self.use_set:
                    r_tp += int(t in pred_args)
                else:
                    r_tp += min(pred_args[t], v)
            # if self.count>=600 and len(pred_args) != len(args):
            #     import pdb
            #     pdb.set_trace()
            self.r_tp += r_tp
            if self.use_set:
                self.r_pre += len(pred_args)
                self.r_rec += len(args)
            else:
                self.r_pre += sum(pred_args.values())
                self.r_rec += sum(args.values())

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        f, rec, pre = _compute_f_rec_pre(self.r_tp, self.r_rec, self.r_pre)
        res.update({'r_f': f, 'r_rec': rec, 'r_pre': pre})
        return res


class EEMetric_(Metric):
    def __init__(self, matrix_segs, tri_thres, role_thres, arg_thres, allow_nested=False,
                 constrain=None, symmetric=False, use_set=True):
        # constrain: {key: set()}  其中key为event_type的index，value是这个event_type运行的role有哪些
        super(EEMetric_, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        self.register_element('r_tp', 0, aggregate_method='sum')
        self.register_element('r_pre', 0, aggregate_method='sum')
        self.register_element('r_rec', 0, aggregate_method='sum')

        self.num_tri_label = matrix_segs['tri']
        self.num_role_label = matrix_segs['role']
        self.role_thres = role_thres
        self.tri_thres = tri_thres
        self.arg_thres = arg_thres
        self.allow_nested = allow_nested
        self.count = 0
        self.constrain = constrain
        self.symmetric = symmetric
        self.use_set = use_set

    def update(self, tri_target, arg_target, scores, word_len):
        assert self.num_tri_label + self.num_role_label == scores.size(-1) - 1
        if self.symmetric is True:
            scores[..., :self.num_tri_label + 1] = (scores[..., :self.num_tri_label + 1] +
                                                    scores[..., :self.num_tri_label + 1].transpose(1, 2)) / 2

        arg_logits = scores[..., 0]  # bsz x max_len x max_len
        tri_logits = scores[..., 1:self.num_tri_label + 1]
        role_logits = scores[..., self.num_tri_label + 1:].cpu().numpy()
        tri_pred = tri_logits.max(dim=-1)[0]

        # [set([(s, e, r), ()]), set()]
        if self.symmetric == 2:
            tri_preds = symmetric_decode(tri_pred, word_len, allow_nested=self.allow_nested, thres=self.tri_thres)
        else:
            tri_preds = decode(tri_pred, word_len, allow_nested=self.allow_nested, thres=self.tri_thres)
        # arg_preds = arg_logits>=self.arg_thres
        arg_preds = decode(arg_logits, word_len, allow_nested=True, thres=self.arg_thres)
        self.count += 1
        for ins_idx, (tris, args, tri_pred, tri_score, role_score) in enumerate(zip(tri_target, arg_target, tri_preds,
                                                                                    tri_logits.cpu().numpy(),
                                                                                    role_logits)):
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
            arg_pred = [(s, e) for s, e, t in arg_preds[ins_idx]]  # [(s, e), (s, e)]
            # 这里如果，arg_pred没有和任何trigger的分数大于某个值，那说明这个arg是不需要的
            pred_args = Counter()
            for i in range(len(pred_tris)):
                for j in range(len(arg_pred)):
                    s, e, t = pred_tris[i]
                    a_s, a_e = arg_pred[j]
                    hh_score = role_score[s, a_s]  # num_rel
                    tt_score = role_score[e, a_e]
                    score = (hh_score + tt_score) / 2
                    # pred_roles = score >= self.role_thres
                    if score.max() < self.role_thres:
                        continue
                    idx = score.argmax()
                    if idx in self.constrain[t]:
                        pred_args[(a_s, a_e, idx, t)] += 1
            r_tp = 0
            args = Counter(map(tuple, args))
            for t, v in args.items():
                if self.use_set:
                    r_tp += int(t in pred_args)
                else:
                    r_tp += min(pred_args[t], v)
            # if self.count>=600 and len(pred_args) != len(args):
            #     import pdb
            #     pdb.set_trace()
            self.r_tp += r_tp
            if self.use_set:
                self.r_pre += len(pred_args)
                self.r_rec += len(args)
            else:
                self.r_pre += sum(pred_args.values())
                self.r_rec += sum(args.values())

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        f, rec, pre = _compute_f_rec_pre(self.r_tp, self.r_rec, self.r_pre)
        res.update({'r_f': f, 'r_rec': rec, 'r_pre': pre})
        return res
