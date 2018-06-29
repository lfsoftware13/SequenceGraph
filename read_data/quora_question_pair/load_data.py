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
        # df = df.drop_duplicates()
        size = len(df)
        test = df.sample(n=int(size*0.1), )
        df = df.drop(test.index)
        valid = df.sample(n=int(size*0.1), )
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


@disk_cache(basename='quora.load_character_vocabulary', directory=CACHE_DATA_PATH)
def load_character_vocabulary(n_gram):
    from read_data import character_embedding
    word_set = load_word_set()
    character_vocabulary = character_embedding.load_character_vocabulary(n_gram=n_gram, token_list=word_set)
    return character_vocabulary


def map_character(character_vocabulary, df: pd.DataFrame):
    df['parsed_character_question1'] = character_vocabulary.parse_string_without_padding(df['tokenized_question1'])
    df['parsed_character_question2'] = character_vocabulary.parse_string_without_padding(df['tokenized_question2'])
    return df


@disk_cache(basename='quora.load_parsed_data_with_character_embedding', directory=CACHE_DATA_PATH)
def load_parsed_data_with_character_embedding(debug=False, word_vector_name="fasttext", n_gram=1):
    character_vocabulary = load_character_vocabulary(n_gram=n_gram)
    train, valid, test = [map_character(character_vocabulary, t) for t in load_parsed_data(debug=debug, word_vector_name=word_vector_name)]
    return train, valid, test


# @disk_cache(basename='quora.load_parsed_quora_data', directory=CACHE_DATA_PATH)
def load_parsed_quora_data(debug=False, word_vector_name="fasttext", n_gram=1):

    def to_dataset(df: pd.DataFrame):
        res = []
        for row in df.iterrows():
            row = row[1]
            if len(row['tokenized_question1']) + len(row['tokenized_question2']) > 100:
                continue
            res.append(
                {
                    "q1_str": row['tokenized_question1'],
                    "q2_str": row['tokenized_question2'],
                    "s1": row['parsed_question1'],
                    "s2": row['parsed_question2'],
                    "s1_char": row['parsed_character_question1'],
                    "s2_char": row['parsed_character_question2'],
                    "label": row['is_duplicate'],
                }
            )
        return res

    from read_data.data_set import OriDataSet
    train, valid, test = [OriDataSet(to_dataset(t)) for t in
                          load_parsed_data_with_character_embedding(debug=debug,
                                                                    word_vector_name=word_vector_name,
                                                                    n_gram=n_gram)]
    vocabulary = load_quora_vocabulary(word_vector_name="fasttext")
    character_vocabulary = load_character_vocabulary(n_gram)
    return train, valid, test, \
           vocabulary.embedding_matrix, len(character_vocabulary.character_to_id_dict),\
           vocabulary.word_to_id(vocabulary.pad), character_vocabulary.character_to_id_dict[character_vocabulary.PAD]


if __name__ == '__main__':
    train, valid, test = load_parsed_data(debug=False, word_vector_name="fasttext")
    q1_length = train['parsed_question1'].map(lambda x: len(x))
    q2_length = train['parsed_question2'].map(lambda x: len(x))
    sum_length = [t1+t2 for t1, t2 in zip(q1_length, q2_length)]
    print("train size:{}, valid size:{},test size:{}".format(len(train), len(valid), len(test)))
    print("max_length:{}".format(max(sum_length)))
    print("the train set size:{}".format(len(sum_length)))
    print("The train set less 200:{}".format(len(list(filter(lambda x: x<200, sum_length)))))
    print("The train set less 100:{}".format(len(list(filter(lambda x: x<100, sum_length)))))
    load_character_vocabulary(n_gram=1)
    load_parsed_data_with_character_embedding(debug=False, word_vector_name="fasttext", n_gram=1)
    load_parsed_quora_data(debug=False, word_vector_name="fasttext", n_gram=1)
