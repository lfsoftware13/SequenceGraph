import more_itertools

from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from read_data import character_embedding
from read_data.pretrained_word_embedding import load_vocabulary
from read_data.snli.read_snli_data import read_snli_split_train_data


@disk_cache(basename='snli.get_snli_token_vocabulary_set', directory=CACHE_DATA_PATH)
def get_snli_token_vocabulary_set():
    train_df = read_snli_split_train_data()
    tokens1 = [tok for tok in train_df['tokens1']]
    tokens2 = [tok for tok in train_df['tokens2']]
    return set(more_itertools.collapse(tokens1)) | set(more_itertools.collapse(tokens2))


@disk_cache(basename='snli.load_snli_vocabulary', directory=CACHE_DATA_PATH)
def load_snli_vocabulary(word_vector_name="fasttext"):
    begin_tokens = ['<BEGIN1>', '<BEGIN2>']
    end_tokens = ['<END1>', '<END2>']
    vocab = load_vocabulary(word_vector_name=word_vector_name, text_list=get_snli_token_vocabulary_set(), use_position_label=True, begin_tokens=begin_tokens, end_tokens=end_tokens)
    return vocab


@disk_cache(basename='snli.load_snli_character_vocabulary', directory=CACHE_DATA_PATH)
def load_snli_character_vocabulary(n_gram):
    word_set = get_snli_token_vocabulary_set()
    character_vocabulary = character_embedding.load_character_vocabulary(n_gram=n_gram, token_list=word_set)
    return character_vocabulary
