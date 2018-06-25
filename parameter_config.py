from torch import optim, nn

from common.evaluate_util import SequenceExactMatch, SequenceOutputIDToWord, SequenceF1Score
from common.problem_util import get_gpu_index
from common.torch_util import calculate_accuracy_of_code_completion
from common.util import PaddedList
from read_data.data_set import OriDataSet
from model.encoder_decoder_graph import EncoderDecoderModel, PreprocessWrapper, EncoderDecoderModelWithPairInput

import pandas as pd

from read_data.method_naming.load_vocabulary import load_summarization_method_name_vocabulary


def em_loss_fn(ignore_token=None,):
    def loss(log_probs, target):
        return calculate_accuracy_of_code_completion(log_probs, target, ignore_token=ignore_token, topk_range=(1, 1),
                                                     gpu_index=get_gpu_index())[1]
    return loss


def NCE_train_loss():
    Loss = nn.CrossEntropyLoss()

    def loss(log_probs, target):
        return Loss(log_probs.permute(0, 2, 1), target)

    return loss


def method_name_config1(is_debug, output_log=None):
    from read_data.method_naming.read_experiment_data import load_method_naming_data
    train, valid, test, embedding_size, pad_id, unk_id, begin_id, hole_id, output_size, output_pad_id = \
        load_method_naming_data(12, is_debug=is_debug, max_token_length=200)
    return {
        "model_fn": EncoderDecoderModel,
        "model_dict": {
            "embedding_dim": 100,
            "n_filters": 200,
            "kernel_size": 5,
            "padding": 2,
            "in_vocabulary_size": embedding_size,
            "hidden_size": 32,
            "out_vocabulary_size": output_size,
            "dynamic_graph_n_layer": 2,
            "graph_attention_n_head": 8,
            "graph_itr": 5,
            "leaky_alpha": 0.2,
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": pad_id,
            "begin_idx": begin_id,
            "hole_idx": hole_id,
        },
        "data": [train, valid, test],
        "batch_size": 6,
        "train_loss": NCE_train_loss,
        "clip_norm": None,
        "name": "sequence_graph_encoder_decoder_for_method_name",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.8, 0.999), "weight_decay": 3e-7, },
        "epcohes": 20,
        "lr": 1e-4,
        "evaluate_object_list": [SequenceExactMatch(ignore_token=output_pad_id, ),
                                 SequenceF1Score(vocab=load_summarization_method_name_vocabulary(), ),
                                 SequenceOutputIDToWord(vocab=load_summarization_method_name_vocabulary(),
                                                        ignore_token=output_pad_id, file_path=output_log, )],
    }


def method_name_config2(is_debug):
    from read_data.method_naming.read_experiment_data import load_method_naming_data
    train, valid, test, embedding_size, pad_id, unk_id, begin_id, hole_id, output_size, output_pad_id = \
        load_method_naming_data(12, is_debug=is_debug, max_token_length=200)
    return {
        "model_fn": EncoderDecoderModelWithPairInput,
        "model_dict": {
            "embedding_dim": 100,
            "n_filters": 200,
            "kernel_size": 5,
            "padding": 2,
            "in_vocabulary_size": embedding_size,
            "hidden_size": 32,
            "out_vocabulary_size": output_size,
            "dynamic_graph_n_layer": 2,
            "graph_attention_n_head": 8,
            "graph_itr": 5,
            "leaky_alpha": 0.2,
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": pad_id,
            "begin_idx": begin_id,
            "hole_idx": hole_id,
        },
        "data": [train, valid, test],
        "batch_size": 8,
        "train_loss": NCE_train_loss,
        "clip_norm": None,
        "name": "sequence_graph_encoder_decoder_for_method_name_with_pair_input",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.8, 0.999), "weight_decay": 3e-7, },
        "epcohes": 20,
        "lr": 1e-4,
        "evaluate_object_list": [SequenceExactMatch(ignore_token=output_pad_id,)],
    }