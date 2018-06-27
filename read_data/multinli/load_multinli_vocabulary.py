import more_itertools

from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from read_data.multinli.read_multinli_data import read_multinli_train_valid_data, \
    read_multinli_split_train_and_valid_data
from read_data.pretrained_word_embedding import load_vocabulary


@disk_cache(basename='get_multinli_token_vocabulary_set', directory=CACHE_DATA_PATH)
def get_multinli_token_vocabulary_set():
    train_df, _ = read_multinli_split_train_and_valid_data()
    tokens1 = [tok for tok in train_df['tokens1']]
    tokens2 = [tok for tok in train_df['tokens2']]
    return set(more_itertools.collapse(tokens1)) | set(more_itertools.collapse(tokens2))


@disk_cache(basename='load_multinli_vocabulary', directory=CACHE_DATA_PATH)
def load_multinli_vocabulary(word_vector_name="fasttext"):
    begin_tokens = ['<BEGIN1>', '<BEGIN2>']
    end_tokens = ['<END1>', '<END2>']
    vocab = load_vocabulary(word_vector_name=word_vector_name, text_list=get_multinli_token_vocabulary_set(), use_position_label=True, begin_tokens=begin_tokens, end_tokens=end_tokens)
    return vocab
