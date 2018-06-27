from abc import abstractmethod, ABCMeta

import torch


class Evaluator(metaclass=ABCMeta):
    @abstractmethod
    def clear_result(self):
        pass

    @abstractmethod
    def add_result(self, log_probs, target, ignore_token, gpu_index, batch_data):
        """

        :param log_probs: [batch, ..., vocab_size]
        :param target: [batch, ...], LongTensor, padded with target token. target.shape == log_probs.shape[:-1]
        :param ignore_token: optional, you can choose special ignore token and gpu index for one batch.
                            or use global value when ignore token and gpu_index is None
        :param gpu_index:
        :return:
        """
        pass

    @abstractmethod
    def get_result(self):
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

    def add_result(self, log_probs, target, ignore_token=None, gpu_index=None, batch_data=None):
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

        if gpu_index is None:
            log_probs = log_probs.cpu()
            target = target.cpu()

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

    def get_result(self):
        return self.match_count / self.batch_count

    def __str__(self):
        return ' SequenceExactMatch top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class SequenceF1Score(Evaluator):
    """
    F1 score evaluator using in paper (A Convolutional Attention Network for Extreme Summarization of Source Code)
    """

    def __init__(self, vocab, rank=1):
        """
        Precision = TP/TP+FP
        Recall = TP/TP+FN
        F1 Score = 2*(Recall * Precision) / (Recall + Precision)
        :param rank: default 1
        :param ignore_token:
        :param gpu_index:
        """
        self.vocab = vocab
        self.rank = rank
        self.tp_count = 0
        # predict_y = TP + FP
        # actual_y = TP + FN
        self.predict_y = 0
        self.actual_y = 0

    def add_result(self, log_probs, target, ignore_token=None, gpu_index=None, batch_data=None):
        """

        :param log_probs: must be 3 dim. [batch, sequence, vocab_size]
        :param target:
        :param ignore_token:
        :param gpu_index:
        :param batch_data:
        :return:
        """
        if isinstance(target, torch.Tensor):
            target = target.cpu()
            target = target.view(target.shape[0], -1)
            target = target.tolist()

        log_probs = log_probs.cpu()
        log_probs = log_probs.view(log_probs.shape[0], -1, log_probs.shape[-1])
        _, top_ids = torch.max(log_probs, dim=-1)
        top_ids = top_ids.tolist()

        end_id = self.vocab.word_to_id(self.vocab.end_tokens[0])
        unk_id = self.vocab.word_to_id(self.vocab.unk)
        batch_tp_count = 0
        batch_predict_y = 0
        batch_actual_y = 0
        for one_predict, one_target in zip(top_ids, target):
            one_predict, _ = self.filter_token_ids(one_predict, end_id, unk_id)
            one_target, _ = self.filter_token_ids(one_target, end_id, unk_id)
            one_tp = set(one_predict) & set(one_target)
            batch_tp_count += len(one_tp)
            batch_predict_y += len(one_predict)
            batch_actual_y += len(one_target)
        self.tp_count += batch_tp_count
        self.predict_y += batch_predict_y
        self.actual_y += batch_actual_y
        precision = float(batch_tp_count ) / float(batch_predict_y)
        recall = float(batch_tp_count) / float(batch_actual_y)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.
        return f1

    def filter_token_ids(self, token_ids, end, unk):
        def filter_special_token(token_ids, val):
            return list(filter(lambda x: x != val, token_ids))
        try:
            end_position = token_ids.index(end)
            token_ids = token_ids[:end_position+1]
        except ValueError as e:
            end_position = None
        token_ids = filter_special_token(token_ids, unk)
        return token_ids, end_position

    def clear_result(self):
        self.tp_count = 0
        self.predict_y = 0
        self.actual_y = 0

    def get_result(self):
        precision = float(self.tp_count) / float(self.predict_y)
        recall = float(self.tp_count) / float(self.actual_y)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.
        return f1

    def __str__(self):
        return ' SequenceF1Score top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class SequenceOutputIDToWord(Evaluator):
    def __init__(self, vocab, ignore_token=None, file_path=None):
        self.vocab = vocab
        self.ignore_token = ignore_token
        self.file_path = file_path
        if file_path is not None:
            with open(file_path, 'w') as f:
                pass

    def add_result(self, log_probs, target, ignore_token=None, gpu_index=None, batch_data=None):
        """

        :param log_probs: [batch, seq, vocab_size]
        :param target: [batch, seq]
        :param ignore_token:
        :param gpu_index:
        :param batch_data:
        :return:
        """
        if self.file_path is None:
            return
        if isinstance(target, torch.Tensor):
            target = target.cpu()
            target = target.tolist()

        log_probs = log_probs.cpu()
        _, top_ids = torch.max(log_probs, dim=-1)
        top_ids = top_ids.tolist()

        input_text = batch_data["text"]

        for one_input, one_top_id, one_target in zip(input_text, top_ids, target):
            predict_token = self.convert_one_token_ids_to_code(one_top_id, self.vocab.id_to_word)
            target_token = self.convert_one_token_ids_to_code(one_target, self.vocab.id_to_word)
            self.save_to_file(one_input, predict_token, target_token)

    def save_to_file(self, input_token=None, predict_token=None, target_token=None):
        if self.file_path is not None:
            with open(self.file_path, 'a') as f:
                f.write('---------------------------------------- one record ----------------------------------------\n')
                if input_token is not None:
                    f.write('input: \n')
                    f.write(str(input_token) + '\n')
                if predict_token is not None:
                    f.write('predict: \n')
                    f.write(predict_token + '\n')
                if target_token is not None:
                    f.write('target: \n')
                    f.write(target_token + '\n')

    def filter_token_ids(self, token_ids, start, end, unk):

        def filter_special_token(token_ids, val):
            return list(filter(lambda x: x != val, token_ids))

        try:
            end_position = token_ids.index(end)
            token_ids = token_ids[:end_position+1]
        except ValueError as e:
            end_position = None
        # token_ids = filter_special_token(token_ids, start)
        # token_ids = filter_special_token(token_ids, end)
        token_ids = filter_special_token(token_ids, unk)
        return token_ids, end_position

    def convert_one_token_ids_to_code(self, token_ids, id_to_word_fn):
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)
        # token_ids, _ = self.filter_token_ids(token_ids, start, end, unk)
        tokens = [id_to_word_fn(tok) for tok in token_ids]
        code = ', '.join(tokens)
        return code

    def clear_result(self):
        pass

    def get_result(self):
        pass

    def __str__(self):
        return ''

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    em_eval = SequenceExactMatch(ignore_token=-1, gpu_index=None)

    log_probs = torch.Tensor([
        [[0.1, 0.3], [0.2, 0.1], [0.4, 0.3], [0.6, 0.8], [0.2, 0.3]],
        [[0.2, 0.1], [0.3, 0.4], [0.5, 0.2], [0.7, 0.8], [0.8, 0.9]]
    ]).cuda(0)
    target = torch.LongTensor([
        [0, 0, 0, 1, -1],
        [0, 1, 0, -1, -1]
    ]).cuda(0)
    part = em_eval.add_result(log_probs, target)
    part = em_eval.add_result(log_probs, target)
    em_eval.clear_result()
    part = em_eval.add_result(log_probs, target)
    print(part)
    log_probs = torch.Tensor([
        [[0.1, 0.3], [0.2, 0.1], [0.4, 0.3], [0.6, 0.8], [0.2, 0.3]],
        [[0.2, 0.1], [0.3, 0.4], [0.5, 0.2], [0.7, 0.8], [0.8, 0.9]]
    ]).cuda(0)
    target = torch.LongTensor([
        [1, 0, 0, 1, -1],
        [0, 1, 0, -1, -1]
    ]).cuda(0)
    part = em_eval.add_result(log_probs, target)
    print(part)

    print(em_eval.match_count, em_eval.batch_count)
    print(em_eval.get_result())


class SequenceBinaryClassExactMatch(Evaluator):

    def __init__(self, rank=1, ignore_token=None, gpu_index=None):
        self.rank = rank
        self.batch_count = 0
        self.match_count = 0
        self.ignore_token = ignore_token
        self.gpu_index = gpu_index

    def clear_result(self):
        self.batch_count = 0
        self.match_count = 0

    def add_result(self, log_probs, target, ignore_token=None, gpu_index=None, batch_data=None):
        """

        :param log_probs: [batch, ...]
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

        if gpu_index is None:
            log_probs = log_probs.cpu()
            target = target.cpu()

        top1_id = torch.gt(log_probs, 0.5).long()
        # top1_id = torch.squeeze(top1_id, dim=-1)

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

    def get_result(self):
        if self.batch_count == 0:
            return 0
        return self.match_count / self.batch_count

    def __str__(self):
        return ' SequenceBinaryClassExactMatch top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()