import math

from fastNLP import seq_len_to_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from transformers import AutoModel

from model.args import ARGS
from model.cross_transformer import CrossTransformer


class UnifyModel(nn.Module):
    def __init__(self, model_name, matrix_segs, use_at_loss=True, cross_dim=200, cross_depth=3,
                 biaffine_size=200, use_ln=False, drop_s1_p=0.1, use_s2=False, empty_rel_weight=0.1,
                 attn_dropout=0.15, use_tri_bias=True):
        super(UnifyModel, self).__init__()
        self.matrix_segs = matrix_segs
        self.drop_s1_p = drop_s1_p
        self.use_s2 = use_s2
        self.empty_rel_weight = empty_rel_weight
        if ARGS.get('hidden_dropout', None) is not None:
            self.pretrain_model = AutoModel.from_pretrained(model_name, hidden_dropout_prob=ARGS['hidden_dropout'])
        else:
            self.pretrain_model = AutoModel.from_pretrained(model_name)

        hidden_size = self.pretrain_model.config.hidden_size
        self.hidden_size = hidden_size
        if ARGS.get('use_pos', False) is True:
            self.pos_embed = nn.Parameter(nn.init.xavier_normal_(torch.randn(512, biaffine_size), gain=0.1))
        elif ARGS.get('use_pos', False) in ('span', 'span2', 'span3'):
            n_pos = 300
            self.span_embed = torch.nn.Embedding(n_pos, cross_dim)
            if ARGS.get('use_pos', False) == 'span3':
                nn.init.xavier_normal_(self.span_embed.weight.data, gain=0.1)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
            self.register_buffer('span_size_ids', _span_size_ids.long())

        hsz = biaffine_size * 2 + 2
        if ARGS.get('use_size_embed', False) is 1:
            n_pos = 50
            self.size_embedding = torch.nn.Embedding(n_pos, ARGS.get('size_embed_dim', 25))
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz += ARGS.get('size_embed_dim', 25)
        if ARGS.get('use_size_embed', False) is 2:
            n_pos = 50
            self.length_embed = torch.nn.Embedding(n_pos, cross_dim)
            # torch.nn.init.xavier_normal_(self.length_embed.weight.data, gain=0.1)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
            self.register_buffer('span_size_ids', _span_size_ids.long())

        self.biaffine_size = biaffine_size
        self.head_mlp = nn.Sequential(
            nn.Dropout(ARGS.get('drop_p', 0.4)),
            nn.Linear(hidden_size, biaffine_size),
            nn.GELU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(ARGS.get('drop_p', 0.4)),
            nn.Linear(hidden_size, biaffine_size),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(ARGS.get('drop_p', 0.4))
        self.U = nn.Parameter(torch.randn(cross_dim, biaffine_size + 1, biaffine_size + 1))
        torch.nn.init.xavier_normal_(self.U.data)
        if self.use_s2:
            self.W = torch.nn.Parameter(torch.empty(cross_dim, hsz))
            torch.nn.init.xavier_normal_(self.W.data)

        layers = []
        if cross_depth > 0:
            for i in range(cross_depth):
                layers.append(CrossTransformer(dim=cross_dim, dropout=attn_dropout, use_tri_bias=use_tri_bias,
                                               scale=use_ln > 0 or ARGS.get('s1_scale_plus')))
            self.cross_layers = nn.Sequential(*layers)
        if len(matrix_segs) < 5:
            self.down_fc = nn.Linear(cross_dim,
                                     sum(matrix_segs.values()) + 1 if use_at_loss else sum(matrix_segs.values()))
        else:
            self.down_fc = nn.Linear(cross_dim,
                                     sum(matrix_segs.values()) + 2 if use_at_loss else sum(matrix_segs.values()))

        self.use_at_loss = use_at_loss
        if use_at_loss:
            self.at_loss = ATLoss()
        self.use_ln = use_ln
        if use_ln:
            self.layer_norm1 = nn.LayerNorm(cross_dim)
            self.layer_norm2 = nn.LayerNorm(cross_dim)

    def forward(self, input_ids, bpe_len, indexes, matrix):
        # input_ids: bsz x max_len
        # bpe_len: bsz
        # indexes: bsz x max_len
        # matrix: bsz x max_len' x max_len' x n_logits

        attention_mask = seq_len_to_mask(bpe_len)  # bsz x length
        outputs = self.pretrain_model(input_ids, attention_mask=attention_mask.float(), return_dict=True)
        last_hidden_states = outputs['last_hidden_state']

        last_hidden_states = self.dropout(last_hidden_states)

        state = scatter_max(last_hidden_states, index=indexes, dim=1)[0][:, 1:]  # bsz x word_len x hidden_size
        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        if ARGS.get('use_pos', False) is True:
            pos_embed = self.pos_embed[:head_state.size(1)][None]
            head_state += pos_embed
            tail_state += pos_embed

        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        scores1 = torch.einsum('bxi, oij, byj -> bxyo', head_state, self.U, tail_state)
        if self.use_ln == 1:
            scores1 = self.layer_norm1(scores1)
        if self.use_s2 > 0:
            affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                     self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)],
                                    dim=-1)
            if hasattr(self, 'size_embedding'):
                size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
                affined_cat = torch.cat([affined_cat,
                                         self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)],
                                        dim=-1)
            scores2 = torch.einsum('bmnh,kh->bmnk', affined_cat, self.W)
            if self.use_ln == 1:
                scores2 = self.layer_norm2(scores2)
            if self.use_s2 == 1:
                scores = scores1 + scores2
        else:
            scores = scores1

        if ARGS.get('use_pos', False) == 'span':
            span_ids = self.span_size_ids[:head_state.size(1), :head_state.size(1)]
            span_embed = self.span_embed(span_ids)[None]  # 1 x max_len x max_len x biaffine_size
            scores = scores + span_embed
        if ARGS.get('use_size_embed', False) is 2:
            span_ids = self.span_size_ids[:head_state.size(1), :head_state.size(1)]
            span_embed = self.dropout(self.length_embed(span_ids))[None]
            scores = scores + span_embed

        if hasattr(self, 'cross_layers'):
            lengths, _ = indexes.max(dim=-1)
            mask = seq_len_to_mask(lengths)
            mask = mask[:, None] * mask.unsqueeze(-1)  # bsz x length x length
            pad_mask = mask[..., None].eq(0)
            if ARGS.get('use_pos', False) == 'span2':
                span_ids = self.span_size_ids[:head_state.size(1), :head_state.size(1)]
                span_embed = self.span_embed(span_ids)[None]  # 1 x max_len x max_len x biaffine_size
                scores1 = scores1 + span_embed
            if ARGS.get('s1_scale_plus', False):
                scores1 = scores1 / math.sqrt(self.biaffine_size)
            if ARGS.get('use_pos', False) == 'span3':
                span_ids = self.span_size_ids[:head_state.size(1), :head_state.size(1)]
                span_embed = self.span_embed(span_ids)[None]  # 1 x max_len x max_len x biaffine_size
                scores1 = scores1 + span_embed
            if self.use_ln == 2:
                scores1 = self.layer_norm1(scores1)
                if self.use_s2:
                    scores2 = self.layer_norm2(scores2)
            if self.training and self.drop_s1_p > 0:
                scores1 = F.dropout2d(scores1.permute(0, 3, 1, 2), p=self.drop_s1_p).permute(0, 2, 3, 1)
            if self.use_s2 > 0:
                u_scores = (scores1 + scores2).masked_fill(pad_mask, 0)  # bsz x max_len x max_len x dim
            else:
                u_scores = (scores1).masked_fill(pad_mask, 0)  # bsz x max_len x max_len x dim
            u_scores = self.cross_layers((u_scores, pad_mask.squeeze(-1).contiguous()))[0]
            if ARGS.get('use_residual', True):
                scores = u_scores + scores
            else:
                scores = u_scores
        scores = self.down_fc(scores)
        scores.masked_fill_(scores.isinf() | scores.isnan(), 0)
        assert scores.size(-1) == matrix.size(-1)
        return {'scores': scores}

    def forward_ner(self, input_ids, bpe_len, indexes, matrix):
        outputs = self(input_ids, bpe_len, indexes, matrix)
        if self.training:
            scores = outputs['scores']
            flat_scores = scores.reshape(-1)
            flat_matrix = matrix.reshape(-1)
            mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()
            return {'loss': loss}
        return outputs

    def forward_re(self, input_ids, bpe_len, indexes, matrix, rel_mask):
        outputs = self(input_ids, bpe_len, indexes, matrix)
        scores = outputs['scores']
        if self.training:
            ner_scores = scores[..., :self.matrix_segs['ent']]
            ner_matrix = matrix[..., :self.matrix_segs['ent']]
            ner_scores = ner_scores.reshape(-1)
            ner_matrix = ner_matrix.reshape(-1)
            mask = ner_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(ner_scores, ner_matrix.float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()

            rel_scores = scores[..., self.matrix_segs['ent']:]
            rel_matrix = matrix[..., self.matrix_segs['ent']:]

            if hasattr(self, 'at_loss'):
                rel_loss = self.at_loss(rel_scores, rel_matrix.masked_fill(rel_matrix.eq(-100), 0))
                rel_mask = rel_mask[..., None]
                rel_loss = (rel_loss * rel_mask + rel_loss * rel_mask.eq(0) * self.empty_rel_weight) * (
                    rel_matrix.ne(-100).float())
                loss = rel_loss.reshape(input_ids.size(0), -1).sum(dim=-1).mean() + loss
            else:
                bsz, max_len, max_len, n_rel = rel_scores.size()
                flat_scores = rel_scores.reshape(-1)
                flat_matrix = rel_matrix.reshape(-1)
                mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
                flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
                rel_mask = rel_mask[..., None].float()
                flat_loss = flat_loss.reshape(bsz, max_len, max_len, n_rel)
                flat_loss = flat_loss * rel_mask + flat_loss * rel_mask.eq(0) * self.empty_rel_weight
                # flat_loss = flat_loss.reshape(input_ids.size(0), -1).sum(dim=-1)*
                loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean() + loss

            return {'loss': loss}

        else:
            if self.use_at_loss is True:
                ner_scores = scores[..., :self.matrix_segs['ent']]
                rel_scores = scores[..., self.matrix_segs['ent']:]
                rel_scores = self.at_loss.get_label(rel_scores)[..., 1:]
                outputs['scores'] = torch.cat([ner_scores, rel_scores], dim=-1)
            elif self.use_at_loss == 2:
                scores = self.at_loss.get_label(scores)[..., 1:]
                outputs['scores'] = scores

        return outputs

    def forward_ee(self, input_ids, bpe_len, indexes, matrix, rel_mask):
        outputs = self(input_ids, bpe_len, indexes, matrix)
        scores = outputs['scores']
        if self.training:
            num_ent = self.matrix_segs['tri'] + 1
            ner_scores = scores[..., :num_ent]
            ner_matrix = matrix[..., :num_ent]
            ner_scores = ner_scores.reshape(-1)
            ner_matrix = ner_matrix.reshape(-1)
            mask = ner_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(ner_scores, ner_matrix.float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()

            rel_scores = scores[..., num_ent:]
            rel_matrix = matrix[..., num_ent:]

            if hasattr(self, 'at_loss'):
                rel_loss = self.at_loss(rel_scores, rel_matrix.masked_fill(rel_matrix.eq(-100), 0))
                rel_mask = rel_mask[..., None]
                rel_loss = (rel_loss * rel_mask + rel_loss * rel_mask.eq(0) * self.empty_rel_weight) * (
                    rel_matrix.ne(-100).float())
                loss = rel_loss.reshape(input_ids.size(0), -1).sum(dim=-1).mean() + loss
            else:
                bsz, max_len, max_len, n_rel = rel_scores.size()
                flat_scores = rel_scores.reshape(-1)
                flat_matrix = rel_matrix.reshape(-1)
                mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
                flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
                rel_mask = rel_mask[..., None].float()
                flat_loss = flat_loss.reshape(bsz, max_len, max_len, n_rel)
                flat_loss = flat_loss * rel_mask + flat_loss * rel_mask.eq(0) * self.empty_rel_weight
                # flat_loss = flat_loss.reshape(input_ids.size(0), -1).sum(dim=-1)*
                loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean() + loss

            return {'loss': loss}
        else:
            if self.use_at_loss is True:
                tri_scores = scores[..., :self.matrix_segs['tri'] + 1].sigmoid()
                role_scores = scores[..., self.matrix_segs['tri'] + 1:]
                role_scores = self.at_loss.get_label(role_scores)[..., 1:]
                outputs['scores'] = torch.cat([tri_scores, role_scores], dim=-1)
            else:
                scores[..., :self.matrix_segs['tri'] + 1] = scores[..., :self.matrix_segs['tri'] + 1].sigmoid()
                outputs['scores'] = scores

        return outputs

    def forward_ie(self, input_ids, bpe_len, indexes, matrix, rel_mask):
        outputs = self(input_ids, bpe_len, indexes, matrix)
        scores = outputs['scores']
        at_loss_shift = 1 if hasattr(self, 'at_loss') else 0
        if self.training:
            num_ent = self.matrix_segs['ent']
            ner_scores = scores[..., :num_ent]
            ner_matrix = matrix[..., :num_ent]
            ner_scores = ner_scores.reshape(-1)
            ner_matrix = ner_matrix.reshape(-1)
            mask = ner_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(ner_scores, ner_matrix.float(), reduction='none')
            ent_loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()

            rel_scores = scores[..., num_ent:self.matrix_segs['rel'] + num_ent + at_loss_shift]
            rel_matrix = matrix[..., num_ent:self.matrix_segs['rel'] + num_ent + at_loss_shift]
            if hasattr(self, 'at_loss'):
                rel_loss = self.at_loss(rel_scores, rel_matrix.masked_fill(rel_matrix.eq(-100), 0))
                _rel_mask = rel_mask[..., None]
                rel_loss = (rel_loss * _rel_mask + rel_loss * _rel_mask.eq(0) * self.empty_rel_weight) * (
                    rel_matrix.ne(-100).float())
                rel_loss = rel_loss.reshape(input_ids.size(0), -1).sum(dim=-1).mean()
            else:
                bsz, max_len, max_len, n_rel = rel_scores.size()
                flat_scores = rel_scores.reshape(-1)
                flat_matrix = rel_matrix.reshape(-1)
                mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
                flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
                _rel_mask = rel_mask[..., None].float()
                flat_loss = flat_loss.reshape(bsz, max_len, max_len, n_rel)
                flat_loss = flat_loss * _rel_mask + flat_loss * _rel_mask.eq(0) * self.empty_rel_weight
                # flat_loss = flat_loss.reshape(input_ids.size(0), -1).sum(dim=-1)*
                rel_loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()

            shift = self.matrix_segs['ent'] + self.matrix_segs['rel'] + at_loss_shift  # 有一个是位置shift
            num_tri = self.matrix_segs['tri']
            ner_scores = scores[..., shift:shift + num_tri + 1]
            ner_matrix = matrix[..., shift:shift + num_tri + 1]
            ner_scores = ner_scores.reshape(-1)
            ner_matrix = ner_matrix.reshape(-1)
            mask = ner_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(ner_scores, ner_matrix.float(), reduction='none')
            tri_loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()

            rel_scores = scores[..., shift + num_tri + 1:]
            rel_matrix = matrix[..., shift + num_tri + 1:]
            if hasattr(self, 'at_loss'):
                role_loss = self.at_loss(rel_scores, rel_matrix.masked_fill(rel_matrix.eq(-100), 0))
                _rel_mask = rel_mask[..., None]
                role_loss = (role_loss * _rel_mask + role_loss * _rel_mask.eq(0) * self.empty_rel_weight) * (
                    rel_matrix.ne(-100).float())
                role_loss = role_loss.reshape(input_ids.size(0), -1).sum(dim=-1).mean()
            else:
                bsz, max_len, max_len, n_rel = rel_scores.size()
                flat_scores = rel_scores.reshape(-1)
                flat_matrix = rel_matrix.reshape(-1)
                mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
                flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
                _rel_mask = rel_mask[..., None].float()
                flat_loss = flat_loss.reshape(bsz, max_len, max_len, n_rel)
                flat_loss = flat_loss * _rel_mask + flat_loss * _rel_mask.eq(0) * self.empty_rel_weight
                # flat_loss = flat_loss.reshape(input_ids.size(0), -1).sum(dim=-1)*
                role_loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()

            loss = ent_loss + rel_loss + tri_loss + role_loss
            return {'loss': loss}
        else:
            if self.use_at_loss is True:
                shift = self.matrix_segs['ent'] + self.matrix_segs['rel']
                tri_scores = scores[..., shift + 1:self.matrix_segs['tri'] + 1 + shift + 1].sigmoid()
                role_scores = scores[..., shift + 1 + self.matrix_segs['tri'] + 1:]
                role_scores = self.at_loss.get_label(role_scores)[..., 1:]
                ent_scores = scores[..., :self.matrix_segs['ent']].sigmoid()
                rel_scores = scores[..., self.matrix_segs['ent']:shift + 1]
                rel_scores = self.at_loss.get_label(rel_scores)[..., 1:]
                outputs['scores'] = torch.cat([ent_scores, rel_scores, tri_scores, role_scores], dim=-1)
            else:
                outputs['scores'] = scores.sigmoid()

        return outputs


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[..., 0] = 1.0
        labels[..., 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e6
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e6
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label)

        # Sum two parts
        loss = loss1 + loss2
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[..., :1]
        output = torch.zeros_like(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[..., 0] = (output[..., 1:].sum(-1) == 0.)
        return output
