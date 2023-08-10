import numpy as np

def _compute_f_rec_pre(tp, rec, pre):
    pre = tp/(pre+1e-6)
    rec = tp/(rec+1e-6)
    f = 2*pre*rec/(pre+rec+1e-6)
    return round(f*100, 2), round(rec*100, 2), round(pre*100, 2)


def _spans_from_upper_triangular(seq_len: int):
    """Spans from the upper triangular area.
    """
    return zip(*np.triu_indices(seq_len))
    # for start in range(seq_len):
    #     for end in range(start, seq_len):
    #         yield (start, end)


# def decode(scores, length, allow_nested=False, thres=0.5):
#     batch_chunks = []
#     for idx, (curr_scores, curr_len) in enumerate(zip(scores, length.cpu().tolist())):
#         curr_non_mask = scores.new_ones(curr_len, curr_len, dtype=bool).triu()
#         tmp_scores = curr_scores[:curr_len, :curr_len][curr_non_mask].cpu().numpy()  # -1 x 2
#
#         confidences, label_ids = tmp_scores, tmp_scores>=thres
#         labels = [i for i in label_ids]
#         chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_upper_triangular(curr_len)) if label != 0]
#         confidences = [conf for label, conf in zip(labels, confidences) if label != 0]
#
#         assert len(confidences) == len(chunks)
#         chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
#         chunks = filter_clashed_by_priority(chunks, allow_nested=allow_nested)
#         if len(chunks):
#             batch_chunks.append(set([(s, e, l) for l,s,e in chunks]))
#         else:
#             batch_chunks.append(set())
#     return batch_chunks

def decode(scores, length, allow_nested=False, thres=0.5):
    batch_chunks = []
    for idx, (curr_scores, curr_len) in enumerate(zip(scores, length.cpu().tolist())):
        # curr_non_mask = scores.new_ones(curr_len, curr_len, dtype=bool).triu()
        tmp_scores = curr_scores[:curr_len, :curr_len] # -1 x 2
        tmp = (tmp_scores>=thres).triu()
        chunks = tmp.nonzero(as_tuple=True)  # -1 x 2
        confidences = curr_scores[chunks].tolist()
        chunks = tmp.nonzero()  # -1 x 2
        # import pdb
        # pdb.set_trace()
        assert len(confidences) == len(chunks)
        chunks = [ck for _, ck in sorted(zip(confidences, chunks.tolist()), reverse=True)]
        chunks = filter_clashed_by_priority(chunks, allow_nested=allow_nested)
        if len(chunks):
            batch_chunks.append(set([(s, e, 0) for s, e in chunks]))
        else:
            batch_chunks.append(set())
    return batch_chunks


def symmetric_decode(scores, length, allow_nested=False, thres=0.5):
    batch_chunks = []
    max_len = 256
    matrix = np.arange(max_len*max_len).astype(int).reshape((max_len, max_len))
    assert scores.size(1)<=max_len

    for idx, (curr_scores, curr_len) in enumerate(zip(scores, length.cpu().tolist())):
        # TODO 这里修改为，需要上下三角同时大于才选择出来
        # curr_non_mask = scores.new_ones(curr_len, curr_len, dtype=bool).triu()
        # for upper
        tmp_scores = curr_scores[:curr_len, :curr_len].cpu().numpy()  # -1 x 2
        up_confidences, up_label_ids = tmp_scores, tmp_scores>=thres
        up_indices = np.triu(up_label_ids).nonzero()
        up_chunks = matrix[up_indices]
        up_chunks = [(0, i, j) for i,j in zip(up_chunks//max_len, up_chunks%max_len)]
        up_confidences = up_label_ids[up_indices]

        low_scores = curr_scores.transpose(0, 1)[:curr_len, :curr_len].cpu().numpy()  # -1 x 2
        low_confidences, low_label_ids = low_scores, low_scores>=thres
        # low_chunks = [(label, start, end) for label, (start, end) in zip(low_label_ids, _spans_from_upper_triangular(curr_len)) if label != 0]
        # low_confidences = [conf for label, conf in zip(low_label_ids, low_confidences) if label != 0]
        low_indices = np.triu(low_label_ids).nonzero()
        low_chunks = matrix[low_indices]
        low_chunks = [(0, i, j) for i,j in zip(low_chunks//max_len, low_chunks%max_len)]
        low_confidences = low_label_ids[low_indices]

        chunk2upidx = {t:j for j, t in enumerate(up_chunks)}
        chunks = []
        confidences = []
        for _idx, chunk in enumerate(low_chunks):
            if chunk in chunk2upidx:
                chunks.append(chunk)
                confidences.append(low_confidences[_idx]+up_confidences[chunk2upidx[chunk]])

        assert len(confidences) == len(chunks)
        chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
        chunks = filter_clashed_by_priority(chunks, allow_nested=allow_nested)
        if len(chunks):
            batch_chunks.append(set([(s, e, l) for l,s,e in chunks]))
        else:
            batch_chunks.append(set())
    return batch_chunks


def is_overlapped(chunk1: tuple, chunk2: tuple):
    (s1, e1), (s2, e2) = chunk1, chunk2
    return (s1 < e2 and s2 < e1)


def is_nested(chunk1: tuple, chunk2: tuple):
    (s1, e1), (s2, e2) = chunk1, chunk2
    return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


def is_clashed(chunk1: tuple, chunk2: tuple, allow_nested: bool=True):
    if allow_nested:
        return is_overlapped(chunk1, chunk2) and not is_nested(chunk1, chunk2)
    else:
        return is_overlapped(chunk1, chunk2)


def filter_clashed_by_priority(chunks, allow_nested: bool=True):
    filtered_chunks = []
    for ck in chunks:
        if all(not is_clashed(ck, ex_ck, allow_nested=allow_nested) for ex_ck in filtered_chunks):
            filtered_chunks.append(ck)

    return filtered_chunks
