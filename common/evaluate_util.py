import random

import torch


class Evaluator:
    pass


class SequenceExactMatch(Evaluator):

    def __init__(self, rank=1, ignore_token=None, gpu_index=None):
        self.rank = rank
        self.batch_count = 0
        self.match_count = 0
        self.ignore_token = ignore_token
        self.gpu_index = gpu_index

    def clear_result(self):
        self.batch_count = 0
        self.match_count = 0

    def add_result_top1(self, log_probs, target, ignore_token=None, gpu_index=None):
        """

        :param log_probs: [batch, ..., vocab_size]
        :param target: [batch, ...], LongTensor, padded with target token. target.shape == log_probs.shape[:-1]
        :param ignore_token: optional, you can choose special ignore token and gpu index for one batch.
                            or use global value when ignore token and gpu_index is None
        :param gpu_index:
        :return:
        """
        if ignore_token is None:
            ignore_token = self.ignore_token
        if gpu_index is None:
            gpu_index = self.gpu_index

        if isinstance(target, list):
            target = torch.LongTensor(target)
            if gpu_index is not None:
                target = target.cuda(gpu_index)

        _, top1_id = torch.topk(log_probs, k=1, dim=-1)
        top1_id = torch.squeeze(top1_id, dim=-1)

        not_equal_result = torch.ne(top1_id, target)

        if ignore_token is not None:
            target_mask = torch.ne(target, ignore_token)
            not_equal_result = not_equal_result & target_mask
        batch_error_count = not_equal_result
        for i in range(len(not_equal_result.shape)-1):
            batch_error_count = torch.sum(batch_error_count, dim=-1)

        # [batch]
        batch_result = torch.eq(batch_error_count, 0)
        batch_match_count = torch.sum(batch_result).data.item()

        batch_size = log_probs.shape[0]
        self.batch_count += batch_size
        self.match_count += batch_match_count
        return batch_match_count / batch_size

    def get_top1_result(self):
        return self.match_count / self.batch_count


if __name__ == "__main__":
    em_eval = SequenceExactMatch(ignore_token=-1, gpu_index=0)

    log_probs = torch.Tensor([
        [[0.1, 0.3], [0.2, 0.1], [0.4, 0.3], [0.6, 0.8], [0.2, 0.3]],
        [[0.2, 0.1], [0.3, 0.4], [0.5, 0.2], [0.7, 0.8], [0.8, 0.9]]
    ]).cuda(0)
    target = torch.LongTensor([
        [0, 0, 0, 1, -1],
        [0, 1, 0, -1, -1]
    ]).cuda(0)
    part = em_eval.add_result_top1(log_probs, target)
    part = em_eval.add_result_top1(log_probs, target)
    em_eval.clear_result()
    part = em_eval.add_result_top1(log_probs, target)
    print(part)
    log_probs = torch.Tensor([
        [[0.1, 0.3], [0.2, 0.1], [0.4, 0.3], [0.6, 0.8], [0.2, 0.3]],
        [[0.2, 0.1], [0.3, 0.4], [0.5, 0.2], [0.7, 0.8], [0.8, 0.9]]
    ])
    target = torch.LongTensor([
        [1, 0, 0, 1, -1],
        [0, 1, 0, -1, -1]
    ])
    part = em_eval.add_result_top1(log_probs, target)
    print(part)

    print(em_eval.match_count, em_eval.batch_count)
    print(em_eval.get_top1_result())
