import pandas as pd
import more_itertools

from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from config import Quora_data_set_path
from common import nlp_util
from read_data.pretrained_word_embedding import Vocabulary


def load_quora_data():
    return pd.read_csv(Quora_data_set_path)


@disk_cache(basename='quora.load_tokenized_quora_data', directory=CACHE_DATA_PATH)
def load_tokenized_quora_data(debug=False):
    df = load_quora_data()

    if debug:
        df = df.head(300)

    q1 = df['question1'].map(lambda x: str(x))
    q2 = df['question2'].map(lambda x: str(x))

    q1 = nlp_util.parallel_tokenize(10, nlp_util.stanford_tokenize, q1)
    q2 = nlp_util.parallel_tokenize(10, nlp_util.stanford_tokenize, q2)

    df['tokenized_question1'] = q1
    df['tokenized_question2'] = q2

    return df


def create_word_set(df: pd.DataFrame):
    word_set = set(more_itertools.collapse(df['tokenized_question1']))
    word_set |= set(more_itertools.collapse(df['tokenized_question2']))
    return word_set


@disk_cache(basename='quora.load_word_set', directory=CACHE_DATA_PATH)
def load_word_set():
    train, _, _ = load_split_data(debug=False)
    return create_word_set(train)


@disk_cache(basename='quora.load_quora_vocabulary', directory=CACHE_DATA_PATH)
def load_quora_vocabulary(word_vector_name="fasttext"):
    from read_data.pretrained_word_embedding import load_vocabulary
    word_set = load_word_set()
    vocabulary = load_vocabulary(word_vector_name, word_set, use_position_label=True)
    return vocabulary


@disk_cache(basename='quora.load_split_data', directory=CACHE_DATA_PATH)
def load_split_data(debug=False,):
    df = load_tokenized_quora_data(debug=debug)
    if not debug:
        df = df.drop_duplicates()
        size = len(df)
        test = df.sample(n=size*0.1, )
        df = df.drop(test.index)
        valid = df.sample(n=size*0.1, )
        train = df.drop(valid.index)
    else:
        train = df.head(100)
        df = df.drop(train.index)
        valid = df.head(100)
        test = df.drop(valid.index)
    return train, valid, test


def map_token(df: pd.DataFrame, vocabulary:Vocabulary):
    df['parsed_question1'] = vocabulary.parse_text_without_pad(df['tokenized_question1'])
    df['parsed_question2'] = vocabulary.parse_text_without_pad(df['tokenized_question2'])
    return df


@disk_cache(basename='quora.load_parsed_data', directory=CACHE_DATA_PATH)
def load_parsed_data(debug=False, word_vector_name="fasttext"):
    vocabulary = load_quora_vocabulary(word_vector_name=word_vector_name)
    train, valid, test = [map_token(df, vocabulary) for df in load_split_data(debug=debug)]
    return train, valid, test


if __name__ == '__main__':
    load_parsed_data(debug=False, word_vector_name="fasttext")