from torch import optim, nn

from common.evaluate_util import SequenceExactMatch, SequenceOutputIDToWord, SequenceF1Score, \
    SequenceBinaryClassExactMatch
from common.opt import OpenAIAdam
from common.problem_util import get_gpu_index
from common.schedculer import LinearScheduler
from common.torch_util import calculate_accuracy_of_code_completion
from model.self_attention_model import SelfAttentionPairModel
from model.sentence_pair_graph import GGNNGraphModel, TestModel

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


def BCELoss():
    Loss = nn.BCEWithLogitsLoss()

    def loss(log_probs, target):
        return Loss(log_probs, target.float())

    return loss


def method_name_config1(is_debug, output_log=None):
    from model.encoder_decoder_graph import EncoderDecoderModel, PreprocessWrapper
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
    from model.encoder_decoder_graph import PreprocessWrapper, EncoderDecoderModelWithPairInput
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


def quora_dataset_config1(is_debug, output_log=None):
    from model.sentence_pair_graph import SequenceGraphModel, PreprocessWrapper
    from read_data.quora_question_pair.load_data import load_parsed_quora_data
    train, valid, test, embedding_matrix, character_size, word_pad_id, character_pad_id = \
        load_parsed_quora_data(debug=is_debug,
                               word_vector_name="glove_300d",
                               n_gram=1)
    return {
        "model_fn": SequenceGraphModel,
        "model_dict": {
            "word_embedding": embedding_matrix,
            "character_number": character_size,
            "character_embedding_dim": 16,
            "character_n_filters": 32,
            "character_kernel_size": 5,
            "character_padding": 2,
            "n_link_type": 3,
            "hidden_state_size": 200,
            "n_dynamic_link_layer": 2,
            "n_fix_graph": 1,
            "graph_itr": 5,
            "n_classes": 2,
            "summary_node": False,
            "tie_weight": False,
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": word_pad_id,
            "character_pad_idx": character_pad_id,
            "summary_node": True,
        },
        "data": [train, valid, test],
        "batch_size": 8,
        "train_loss": BCELoss,
        "clip_norm": 10,
        "name": "sequence_graph_encoder_decoder_for_method_name",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.8, 0.999), "weight_decay": 3e-7, },
        "epcohes": 80,
        "lr": 1e-4,
        "evaluate_object_list": [SequenceBinaryClassExactMatch(gpu_index=get_gpu_index())],
    }


def quora_dataset_config2(is_debug, output_log=None):
    from model.sentence_pair_graph import SequenceGraphModelWithGraphAttention, PreprocessWrapper
    from read_data.quora_question_pair.load_data import load_parsed_quora_data
    train, valid, test, embedding_matrix, character_size, word_pad_id, character_pad_id = \
        load_parsed_quora_data(debug=is_debug,
                               word_vector_name="glove_300d",
                               n_gram=1)
    return {
        "model_fn": SequenceGraphModelWithGraphAttention,
        "model_dict": {
            "word_embedding": embedding_matrix,
            "character_number": character_size,
            "mixed": True,
            "character_embedding_dim": 600,
            "character_n_filters": 200,
            "character_kernel_size": 5,
            "character_padding": 2,
            "hidden_size": 128,
            "graph_itr": 6,
            "dynamic_graph_n_layer": 2,
            "graph_attention_n_head": 6,
            "leaky_alpha": 0.2,
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": word_pad_id,
            "character_pad_idx": character_pad_id,
            "summary_node": False,
        },
        "data": [train, valid, test],
        "batch_size": 8,
        "train_loss": BCELoss,
        "clip_norm": 10,
        "name": "sequence_graph_encoder_decoder_for_method_name",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.8, 0.999), "weight_decay": 3e-7, },
        "epcohes": 80,
        "lr": 1e-4,
        "evaluate_object_list": [SequenceBinaryClassExactMatch(gpu_index=get_gpu_index())],
    }


def quora_dataset_config3(is_debug, output_log=None):
    from model.self_attention_model import SelfAttentionPairModel, PreprocessWrapper
    from read_data.quora_question_pair.load_data import load_parsed_quora_data
    train, valid, test, embedding_matrix, character_size, word_pad_id, character_pad_id = \
        load_parsed_quora_data(debug=is_debug,
                               word_vector_name="fasttext",
                               n_gram=1)
    return {
        "model_fn": SelfAttentionPairModel,
        "model_dict": {
            "word_embedding": embedding_matrix,
            "character_number": character_size,
            "character_embedding_dim": 600,
            "character_n_filters": 200,
            "character_kernel_size": 5,
            "character_padding": 2,
            "self_attention_layer": 5,
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": word_pad_id,
            "character_pad_idx": character_pad_id,
        },
        "data": [train, valid, test],
        "batch_size": 8,
        "train_loss": BCELoss,
        "clip_norm": 10,
        "name": "sequence_graph_encoder_decoder_for_method_name",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.8, 0.999), "weight_decay": 3e-7, },
        "epcohes": 80,
        "lr": 1e-4,
        "evaluate_object_list": [SequenceBinaryClassExactMatch(gpu_index=get_gpu_index())],
    }


def quora_dataset_config4(is_debug, output_log=None):
    from model.self_attention_model import PreprocessWrapper
    from model.graph_cluster_model import GraphClusterModel
    from read_data.quora_question_pair.load_data import load_parsed_quora_data
    train, valid, test, embedding_matrix, character_size, word_pad_id, character_pad_id = \
        load_parsed_quora_data(debug=is_debug,
                               word_vector_name="glove_300d",
                               n_gram=1)
    return {
        "model_fn": GraphClusterModel,
        "model_dict": {
            "word_embedding": embedding_matrix,
            "character_number": character_size,
            "character_embedding_dim": 16,
            "character_n_filters": 32,
            "character_kernel_size": 5,
            "character_padding": 2,
            "hidden_size": 128, "conv_type": "depthwise_separable",
            "resize_kernel_size": 7, "resize_pad_size": 3,
            "n_encoder_conv_layer": 2, "encoder_kernel_size": 7, "encoder_padding": 3,
            "n_self_attention_heads": 4, "route_number": 3, "n_capsules": 32,
            "capsules_dim": 128, "n_compare_layer": 2, "n_layer_output_conv": 2,
            "n_layer_output_feedforward": 3, "hidden_size_output_feedforward": 128, "n_classes": 2, "dropout": 0.2
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": word_pad_id,
            "character_pad_idx": character_pad_id,
        },
        "data": [train, valid, test],
        "batch_size": 16,
        "train_loss": BCELoss,
        "clip_norm": None,
        "name": "sequence_graph_encoder_decoder_for_method_name",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.8, 0.999), "weight_decay": 3e-7, },
        "epcohes": 80,
        "lr": 1e-2,
        "evaluate_object_list": [SequenceBinaryClassExactMatch(gpu_index=get_gpu_index())],
    }


def quora_dataset_config5(is_debug, output_log=None):
    from model.self_attention_model import PreprocessWrapper
    from read_data.quora_question_pair.load_data import load_parsed_quora_data
    train, valid, test, embedding_matrix, character_size, word_pad_id, character_pad_id = \
        load_parsed_quora_data(debug=is_debug,
                               word_vector_name="glove_300d",
                               n_gram=1)
    from qanet.qanet import QANet
    return {
        "model_fn": QANet,
        "model_dict": {
            "word_embedding_matrix": embedding_matrix, "char_embedding_matrix": None,
            "params": {

                "word_embed_dim": 300,

                "highway_n_layers": 2,

                "hidden_size": 128,

                "embed_encoder_resize_kernel_size": 7,
                "embed_encoder_resize_pad": 3,

                "embed_encoder_n_blocks": 1,
                "embed_encoder_n_conv": 4,
                "embed_encoder_kernel_size": 7,
                "embed_encoder_pad": 3,
                "embed_encoder_conv_type": "depthwise_separable",
                "embed_encoder_with_self_attn": False,
                "embed_encoder_n_heads": 8,

                "model_encoder_n_blocks": 7,
                "model_encoder_n_conv": 2,
                "model_encoder_kernel_size": 7,
                "model_encoder_pad": 3,
                "model_encoder_conv_type": "depthwise_separable",
                "model_encoder_with_self_attn": False,
                "model_encoder_n_heads": 8,

                "batch_size": 32,
            }
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": word_pad_id,
            "character_pad_idx": character_pad_id,
        },
        "data": [train, valid, test],
        "batch_size": 64,
        "train_loss": BCELoss,
        "clip_norm": None,
        "name": "qa_net",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.8, 0.999), "weight_decay": 3e-7, },
        "epcohes": 80,
        "lr": 0.001,
        "evaluate_object_list": [SequenceBinaryClassExactMatch(gpu_index=get_gpu_index())],
        "epoch_ratio": 0.25,
    }


def quora_dataset_config6(is_debug, output_log=None):
    from model.self_attention_model import PreprocessWrapper
    from read_data.quora_question_pair.load_data import load_parsed_quora_data
    train, valid, test, embedding_matrix, character_size, word_pad_id, character_pad_id = \
        load_parsed_quora_data(debug=is_debug,
                               word_vector_name="glove_300d",
                               n_gram=1)
    from model.feed_forward_network import FFN
    return {
        "model_fn": FFN,
        "model_dict": {
            "word_embedding": embedding_matrix,
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": word_pad_id,
            "character_pad_idx": character_pad_id,
        },
        "data": [train, valid, test],
        "batch_size": 516,
        "train_loss": BCELoss,
        "clip_norm": None,
        "name": "FFN_try",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.9, 0.999), "weight_decay": 3e-7, },
        "epcohes": 80,
        "lr": 1e-3,
        "evaluate_object_list": [SequenceBinaryClassExactMatch(gpu_index=get_gpu_index())],
        "epoch_ratio": 0.25,
    }


def quora_dataset_config7(is_debug, output_log=None):
    from model.self_attention_model import PreprocessWrapper
    from read_data.quora_question_pair.load_data import load_parsed_quora_data
    train, valid, test, embedding_matrix, character_size, word_pad_id, character_pad_id = \
        load_parsed_quora_data(debug=is_debug,
                               word_vector_name="glove_300d",
                               n_gram=1)
    from model.feed_forward_network import FFNWithCrossCompare
    return {
        "model_fn": FFNWithCrossCompare,
        "model_dict": {
            "word_embedding": embedding_matrix,
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": word_pad_id,
            "character_pad_idx": character_pad_id,
        },
        "data": [train, valid, test],
        "batch_size": 128,
        "train_loss": BCELoss,
        "clip_norm": None,
        "name": "FFN_try",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.9, 0.999), "weight_decay": 3e-7, },
        "epcohes": 80,
        "lr": 1e-3,
        "evaluate_object_list": [SequenceBinaryClassExactMatch(gpu_index=get_gpu_index())],
        "epoch_ratio": 0.25,
    }


def snli_config1(is_debug, output_log=None):
    from model.self_attention_model import PreprocessWrapper
    from read_data.snli.read_snli_experiment_data import load_dict_data
    from read_data.snli.load_snli_vocabulary import load_snli_vocabulary, load_snli_character_vocabulary
    train, valid, test = load_dict_data(debug=is_debug, )
    vocabulary = load_snli_vocabulary("glove_300d")
    character_vocabulary = load_snli_character_vocabulary(n_gram=1)
    from model.feed_forward_network import FFN
    return {
        "model_fn": FFN,
        "model_dict": {
            "word_embedding": vocabulary.embedding_matrix,
            "n_classes": 3,
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": vocabulary.word_to_id(vocabulary.pad),
            "character_pad_idx": character_vocabulary.character_to_id_dict[character_vocabulary.PAD],
        },
        "data": [train, valid, test],
        "batch_size": 516,
        "train_loss": nn.CrossEntropyLoss,
        "clip_norm": None,
        "name": "FFN_snli",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.9, 0.999), "weight_decay": 3e-7, },
        "epcohes": 80,
        "lr": 1e-3,
        "evaluate_object_list": [SequenceExactMatch(gpu_index=get_gpu_index())],
        "epoch_ratio": 0.1,
    }


def snli_config2(is_debug, output_log=None):
    from model.self_attention_model import PreprocessWrapper
    from read_data.snli.read_snli_experiment_data import load_dict_data
    from read_data.snli.load_snli_vocabulary import load_snli_vocabulary, load_snli_character_vocabulary
    train, valid, test = load_dict_data(debug=is_debug, )
    vocabulary = load_snli_vocabulary("glove_300d")
    character_vocabulary = load_snli_character_vocabulary(n_gram=1)
    from model.feed_forward_network import FFNWithCrossCompare
    return {
        "model_fn": FFNWithCrossCompare,
        "model_dict": {
            "word_embedding": vocabulary.embedding_matrix,
            "n_classes": 3,
            "hidden_size": 400,
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": vocabulary.word_to_id(vocabulary.pad),
            "character_pad_idx": character_vocabulary.character_to_id_dict[character_vocabulary.PAD],
        },
        "data": [train, valid, test],
        "batch_size": 32,
        "train_loss": nn.CrossEntropyLoss,
        "clip_norm": None,
        "name": "FFN_snli",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.9, 0.999), "weight_decay": 3e-7, },
        "epcohes": 80,
        "lr": 1e-3,
        "evaluate_object_list": [SequenceExactMatch(gpu_index=get_gpu_index())],
        "epoch_ratio": 0.1,
    }


def snli_config3(is_debug, output_log=None):
    from model.self_attention_model import PreprocessWrapper
    from read_data.snli.read_snli_experiment_data import load_dict_data
    from read_data.snli.load_snli_vocabulary import load_snli_vocabulary, load_snli_character_vocabulary
    train, valid, test = load_dict_data(debug=is_debug, )
    vocabulary = load_snli_vocabulary("glove_300d")
    character_vocabulary = load_snli_character_vocabulary(n_gram=1)
    from qanet.qanet import QANet
    return {
        "model_fn": QANet,
        "model_dict": {
            "word_embedding_matrix": vocabulary.embedding_matrix,
            "char_embedding_matrix": None,
            "params": {

                "word_embed_dim": 300,

                "highway_n_layers": 2,

                "hidden_size": 128,

                "embed_encoder_resize_kernel_size": 7,
                "embed_encoder_resize_pad": 3,

                "embed_encoder_n_blocks": 1,
                "embed_encoder_n_conv": 4,
                "embed_encoder_kernel_size": 7,
                "embed_encoder_pad": 3,
                "embed_encoder_conv_type": "depthwise_separable",
                "embed_encoder_with_self_attn": False,
                "embed_encoder_n_heads": 8,

                "model_encoder_n_blocks": 7,
                "model_encoder_n_conv": 2,
                "model_encoder_kernel_size": 7,
                "model_encoder_pad": 3,
                "model_encoder_conv_type": "depthwise_separable",
                "model_encoder_with_self_attn": False,
                "model_encoder_n_heads": 8,

                "batch_size": 128,
            }
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": vocabulary.word_to_id(vocabulary.pad),
            "character_pad_idx": character_vocabulary.character_to_id_dict[character_vocabulary.PAD],
        },
        "data": [train, valid, test],
        "batch_size": 128,
        "train_loss": nn.CrossEntropyLoss,
        "clip_norm": None,
        "name": "QANet_snli",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.9, 0.999), "weight_decay": 3e-7, },
        "epcohes": 80,
        "lr": 1e-5,
        "evaluate_object_list": [SequenceExactMatch(gpu_index=get_gpu_index())],
        "epoch_ratio": 0.1,
    }


def snli_config4(is_debug, output_log=None):
    from model.self_attention_model import PreprocessWrapper
    from read_data.snli.read_snli_experiment_data import load_dict_data
    from read_data.snli.load_snli_vocabulary import load_snli_vocabulary, load_snli_character_vocabulary
    train, valid, test = load_dict_data(debug=is_debug, )
    vocabulary = load_snli_vocabulary("glove_300d")
    character_vocabulary = load_snli_character_vocabulary(n_gram=1)
    return {
        "model_fn": SelfAttentionPairModel,
        "model_dict": {
            "word_embedding": vocabulary.embedding_matrix,
            "character_number": len(character_vocabulary.character_to_id_dict),
            "character_embedding_dim": 32,
            "character_n_filters": 200,
            "character_kernel_size": 5,
            "character_padding": 2,
            "self_attention_layer": 4,
            "n_classes": 3
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": vocabulary.word_to_id(vocabulary.pad),
            "character_pad_idx": character_vocabulary.character_to_id_dict[character_vocabulary.PAD],
        },
        "data": [train, valid, test],
        "batch_size": 8,
        "train_loss": nn.CrossEntropyLoss,
        "clip_norm": None,
        "name": "self_attention_snli_debug",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.9, 0.999), "weight_decay": 3e-7, },
        "epcohes": 160,
        "lr": 3e-5,
        "evaluate_object_list": [SequenceExactMatch(gpu_index=get_gpu_index())],
        "epoch_ratio": 1,
    }


def snli_config5(is_debug, output_log=None):
    from model.sentence_pair_graph import PreprocessWrapper
    from read_data.snli.read_snli_experiment_data import load_dict_data
    from read_data.snli.load_snli_vocabulary import load_snli_vocabulary, load_snli_character_vocabulary
    train, valid, test = load_dict_data(debug=is_debug, )
    vocabulary = load_snli_vocabulary("glove_300d")
    character_vocabulary = load_snli_character_vocabulary(n_gram=1)
    from model.sentence_pair_graph import SequenceGraphModelWithGraphAttention
    return {
        "model_fn": SequenceGraphModelWithGraphAttention,
        "model_dict": {
            "word_embedding": vocabulary.embedding_matrix,
            "character_number": len(character_vocabulary.character_to_id_dict),
            "mixed": True,
            "character_embedding_dim": 600,
            "character_n_filters": 200,
            "character_kernel_size": 5,
            "character_padding": 2,
            "hidden_size": 128,
            "graph_itr": 1,
            "dynamic_graph_n_layer": 2,
            "graph_attention_n_head": 6,
            "leaky_alpha": 0.2,
            "n_classes": 3,
        },
        "pre_process_module_fn": PreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": vocabulary.word_to_id(vocabulary.pad),
            "character_pad_idx": character_vocabulary.character_to_id_dict[character_vocabulary.PAD],
        },
        "data": [train, valid, test],
        "batch_size": 80,
        "train_loss": nn.CrossEntropyLoss,
        "clip_norm": None,
        "name": "SequenceGraphModelWithGraphAttention_snli",
        "optimizer": optim.Adam,
        "need_pad": True,
        "optimizer_dict": {"betas": (0.9, 0.999), "weight_decay": 3e-7, },
        "epcohes": 160,
        "lr": 3e-3,
        "evaluate_object_list": [SequenceExactMatch(gpu_index=get_gpu_index())],
        "epoch_ratio": 0.1,
    }


def snli_config6(is_debug, output_log=None):
    from model.sentence_pair_graph import ConcatPreprocessWrapper
    import numpy as np
    from read_data.snli.read_snli_experiment_data import load_dict_data
    from read_data.snli.load_snli_vocabulary import load_snli_vocabulary, load_snli_character_vocabulary
    train, valid, test = load_dict_data(debug=is_debug, )
    vocabulary = load_snli_vocabulary("glove_300d")
    # character_vocabulary = load_snli_character_vocabulary(n_gram=1)
    delimeter_idx = len(vocabulary.id_to_word_dict)
    summary_idx = len(vocabulary.id_to_word_dict) + 1
    embedding_matrix = vocabulary.embedding_matrix
    embedding_matrix = np.concatenate((embedding_matrix, np.random.randn(2, embedding_matrix.shape[1])), axis=0)
    return {
        "model_fn": GGNNGraphModel,
        "model_dict": {
            "word_embedding": embedding_matrix,
            "max_length": 80,
            "hidden_state_size": 756,
            "n_classes": 3,
        },
        "pre_process_module_fn": ConcatPreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": vocabulary.word_to_id(vocabulary.pad),
            "delimeter_idx": delimeter_idx,
            "summary_node_idx": summary_idx,
            "max_length": 80,
        },
        "data": [train, valid, test],
        "batch_size": 8,
        "train_loss": nn.CrossEntropyLoss,
        "clip_norm": 1,
        "name": "GGNNGraphModel_snli",
        "optimizer": OpenAIAdam,
        "need_pad": True,
        "optimizer_dict": {
                           "schedule": 'warmup_linear',
                           "warmup": 0.002,
                           "t_total": (100//8)*300,
                           "b1": 0.9,
                           "b2": 0.999,
                           "e": 1e-8,
                           "l2": 0.01,
                           "vector_l2": 'store_true',
                           "max_grad_norm": 1},
        "epcohes": 300,
        "lr": 6.25e-5,
        "evaluate_object_list": [SequenceExactMatch(gpu_index=get_gpu_index())],
        "epoch_ratio": 1,
        "scheduler_fn": None
    }


def snli_config7(is_debug, output_log=None):
    from model.sentence_pair_graph import ConcatPreprocessWrapper
    import numpy as np
    from read_data.snli.read_snli_experiment_data import load_dict_data
    from read_data.snli.load_snli_vocabulary import load_snli_vocabulary, load_snli_character_vocabulary
    train, valid, test = load_dict_data(debug=is_debug, )
    vocabulary = load_snli_vocabulary("glove_300d")
    # character_vocabulary = load_snli_character_vocabulary(n_gram=1)
    delimeter_idx = len(vocabulary.id_to_word_dict)
    summary_idx = len(vocabulary.id_to_word_dict) + 1
    embedding_matrix = vocabulary.embedding_matrix
    embedding_matrix = np.concatenate((embedding_matrix, np.random.randn(2, embedding_matrix.shape[1])), axis=0)
    from model.transformer_lm import dotdict
    return {
        "model_fn": TestModel,
        "model_dict": {
            "cfg": dotdict({
                'n_embd': 768,
                'n_head': 1,
                'n_layer': 1,
                'embd_pdrop': 0.1,
                'attn_pdrop': 0.1,
                'resid_pdrop': 0.1,
                'afn': 'gelu',
                'clf_pdrop': 0.1}),
            "clf_token": summary_idx, "vocabulary_size": embedding_matrix.shape[0],
            "n_ctx": 80 + 2
        },
        "pre_process_module_fn": ConcatPreprocessWrapper,
        "pre_process_module_dict": {
            "pad_idx": vocabulary.word_to_id(vocabulary.pad),
            "delimeter_idx": delimeter_idx,
            "summary_node_idx": summary_idx,
            "max_length": 80,
        },
        "data": [train, valid, test],
        "batch_size": 8,
        "train_loss": nn.CrossEntropyLoss,
        "clip_norm": 1,
        "name": "GGNNGraphModel_snli",
        "optimizer": optim.Adam,
        "need_pad": True,
        # "optimizer_dict": {
        #                    "schedule": 'warmup_linear',
        #                    "warmup": 0.002,
        #                    "t_total": (100//8)*300,
        #                    "b1": 0.9,
        #                    "b2": 0.999,
        #                    "e": 1e-8,
        #                    "l2": 0.01,
        #                    "vector_l2": 'store_true',
        #                    "max_grad_norm": 1},
        "optimizer_dict": {"betas": (0.9, 0.999), "weight_decay": 0.01, },
        "epcohes": 300,
        "lr": 6.25e-5,
        "evaluate_object_list": [SequenceExactMatch(gpu_index=get_gpu_index())],
        "epoch_ratio": 1,
        "scheduler_fn": None
    }