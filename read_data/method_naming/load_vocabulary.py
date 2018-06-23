import more_itertools

from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from read_data.method_naming.read_summarization_source_code_to_method_name_data import \
    read_summarization_train_data_with_valid
from vocabulary.word_vocabulary import load_vocabulary


def read_source_code_token():
    train_df, valid_df = read_summarization_train_data_with_valid()
    names = [method_name for method_name in train_df['tokens']]
    return names


def read_method_name_token():
    train_df, valid_df = read_summarization_train_data_with_valid()
    names = [method_name for method_name in train_df['name']]
    return names


@disk_cache(basename='get_summarization_java_code_vocabulary_set', directory=CACHE_DATA_PATH)
def get_summarization_java_code_vocabulary_set():
    tokens = set(more_itertools.collapse(read_source_code_token()))
    return tokens


@disk_cache(basename='get_summarization_java_code_vocabulary_id_map', directory=CACHE_DATA_PATH)
def get_summarization_java_code_vocabulary_id_map():
    word_list = sorted(get_summarization_java_code_vocabulary_set())
    return {word: i for i, word in enumerate(word_list)}


@disk_cache(basename='create_summarization_java_code_vocabulary', directory=CACHE_DATA_PATH)
def create_summarization_java_code_vocabulary(begin_tokens, end_tokens, unk_token, pad_token, addition_tokens=None):
    vocab = load_vocabulary(get_summarization_java_code_vocabulary_set, get_summarization_java_code_vocabulary_id_map, begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token, pad_token=pad_token, addition_tokens=addition_tokens)
    return vocab


@disk_cache(basename='get_summarization_method_name_vocabulary_set', directory=CACHE_DATA_PATH)
def get_summarization_method_name_vocabulary_set():
    method_names = set(more_itertools.collapse(read_method_name_token()))
    return method_names


@disk_cache(basename='get_summarization_method_name_vocabulary_id_map', directory=CACHE_DATA_PATH)
def get_summarization_method_name_vocabulary_id_map():
    word_list = sorted(get_summarization_method_name_vocabulary_set())
    return {word: i for i, word in enumerate(word_list)}


@disk_cache(basename='create_summarization_method_name_vocabulary', directory=CACHE_DATA_PATH)
def create_summarization_method_name_vocabulary(begin_tokens, end_tokens, unk_token, pad_token, addition_tokens=None):
    vocab = load_vocabulary(get_summarization_method_name_vocabulary_set, get_summarization_method_name_vocabulary_id_map, begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token, pad_token=pad_token, addition_tokens=addition_tokens)
    return vocab


@disk_cache(basename='load_summarization_java_code_vocabulary', directory=CACHE_DATA_PATH)
def load_summarization_java_code_vocabulary():
    begin = '<SENTENCE_START>'
    end = '<SENTENCE_END/>'
    unk = '<SENTENCE_UNK>'
    pad_name = '<SENTENCE_PAD>'
    vocab = create_summarization_java_code_vocabulary([begin], [end], unk, pad_name, addition_tokens=None)
    return vocab


@disk_cache(basename='load_summarization_method_name_vocabulary', directory=CACHE_DATA_PATH)
def load_summarization_method_name_vocabulary():
    begin_name = '<METHOD_START>'
    end_name = '<METHOD_END>'
    unk = '<METHOD_UNK>'
    pad_name = '<METHOD_PAD>'
    vocab = create_summarization_method_name_vocabulary([begin_name], [end_name], unk, pad_name, addition_tokens=None)
    return vocab
