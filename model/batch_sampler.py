from itertools import chain

from fastNLP import ReproducibleBatchSampler
import numpy as np


class ConstantTokenBatchSampler(ReproducibleBatchSampler):
    def __init__(self, seq_len, max_token=2000, max_sentence=32, need_be_multiple_of=1, num_bucket=-1,
                 batch_size_when_max_len=2, **kwargs):
        """

        :param List[int] seq_len: list[int], 是每个sample的长度。一般可以通过dataset.get_field('seq_len').content传入
        :param int max_token: 每个batch的最大的token数量
        :param int max_sentence: 每个batch最多多少个instance, -1表示根据max_token决定
        :param int need_be_multiple_of: 生成的batch的instance的数量需要是几的倍数，在DataParallel场景下会用到
        :param int num_bucket: 将数据按长度拆分为num_bucket个bucket，batch中的sample尽量在bucket之中进行组合，这样可以减少padding。
        :param int batch_size_when_max_len: 长度最长的那个batch，最多可以放多少个sample
        """
        super().__init__()
        assert (max_sentence != -1 and max_sentence >= need_be_multiple_of) or max_sentence < 1
        assert len(seq_len) > num_bucket, "The number of samples should be larger than buckets."
        self.seq_len = seq_len
        self.max_token = max_token
        self.max_sentence = max_sentence
        self.batch_size = -1
        self.max_len_in_dataset = max(seq_len)
        self.need_be_multiple_of = need_be_multiple_of
        self.batch_size_when_max_len = batch_size_when_max_len
        seq_len_indice = [(length, i) for i, length in enumerate(seq_len)]
        seq_len_indice.sort(key=lambda x: x[0])
        indice_in_buckets = []
        self.num_bucket = num_bucket
        if num_bucket > 0:
            sample_per_bucket = len(seq_len_indice) // num_bucket
            i = 0
            while len(indice_in_buckets) < len(seq_len_indice):
                indice_in_buckets.append(seq_len_indice[i * sample_per_bucket:(i + 1) * sample_per_bucket])
                i += 1
        else:
            indice_in_buckets = [seq_len_indice]
        self.indice_in_buckets = indice_in_buckets
        self.epoch = 0
        self.rank = kwargs.get('rank', 0)
        self.num_replicas = kwargs.get('num_replicas', 1)
        self.pad = kwargs.get('pad', True)
        self.get_new_order()

    def get_new_order(self):
        rng = np.random.default_rng(self.epoch)
        rng.shuffle(self.indice_in_buckets)
        for bucket in self.indice_in_buckets:
            rng.shuffle(bucket)
        indices = list(chain(*self.indice_in_buckets))
        batches = []
        cur_max_len = 0
        batch = []
        for length, i in indices:
            max_len = max(length, cur_max_len)
            if max_len * (len(batch) + 1) > min(self.max_token,
                                                (
                                                        self.max_len_in_dataset / max_len) ** 3 * self.max_len_in_dataset * self.batch_size_when_max_len) \
                    or len(batch) >= self.max_sentence:
                left_sample = len(batch) % self.need_be_multiple_of
                add_samples = batch.copy()
                cur_max_len = length
                if left_sample != 0:
                    add_samples = add_samples[:-left_sample]
                    batch = batch[-left_sample:]
                    cur_max_len = max(cur_max_len, max([indices[_i][0] for _i in batch]))
                else:
                    batch = []
                if len(add_samples) == 0:
                    raise RuntimeError(
                        f"The sample `{i}` is too long to make a batch with {self.need_be_multiple_of} samples.")
                batches.append(add_samples)
            else:
                cur_max_len = max_len
            batch.append(i)
        if batch:
            left_sample = len(batch) % self.need_be_multiple_of
            add_samples = batch.copy()
            if left_sample != 0:
                add_samples = add_samples[:-left_sample].copy()
            if add_samples:
                batches.append(add_samples)
        rng = np.random.default_rng(self.epoch)
        rng.shuffle(batches)
        # most token 放前面
        # 最长的放前面
        most_token_idx = np.argmax([sum([self.seq_len[b] for b in batch]) for batch in batches])
        most_length_idx = np.argmax(map(len, batches))
        if most_length_idx != most_token_idx:
            for idx in [most_token_idx, most_length_idx]:
                batch = batches.pop(idx)
                batches.insert(0, batch)
        else:
            batch = batches.pop(most_token_idx)
            batches.insert(0, batch)

        self.batches = batches

    def set_distributed(self, num_replicas, rank, pad=True):
        assert num_replicas > 0 and isinstance(num_replicas, int)
        assert isinstance(rank, int) and 0 <= rank < num_replicas
        # 注意初始化该函数时，所有的状态都应当默认是一个 epoch 刚开始训练的状态；
        self.num_replicas = num_replicas
        self.rank = rank
        self.pad = pad
        return self

    def __iter__(self):
        batches = self.batches[self.rank::self.num_replicas]
        if len(self.batches) % self.num_replicas != 0 and len(
                self.batches) % self.num_replicas <= self.rank and self.pad:
            batches = batches + batches[:1]
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.batches) // self.num_replicas) + int(len(self.batches) % self.num_replicas != 0)

    def set_epoch(self, epoch):
        self.epoch = epoch
