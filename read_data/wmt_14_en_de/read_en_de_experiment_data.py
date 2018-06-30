from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from read_data.wmt_14_en_de.load_en_de_vocabulary import load_en_vocabulary, load_de_vocabulary, \
    load_en_character_vocabulary, load_de_character_vocabulary
from read_data.wmt_14_en_de.read_en_de_data import read_en_de_sentence_train_valid_data, read_en_de_sentence_test_data

import numpy as np


@disk_cache(basename='load_wmt14_en_de_data', directory=CACHE_DATA_PATH)
def load_wmt14_en_de_data(is_debug, max_sentence_length=50):
    train_df, valid_df = read_en_de_sentence_train_valid_data()
    test_df = read_en_de_sentence_test_data()
    dfs = [train_df, valid_df, test_df]
    for i in range(len(dfs)):
        print(len(dfs[i]))

    en_vocabulary = load_en_vocabulary()
    de_vocabulary = load_de_vocabulary()
    en_character_vocabulary = load_en_character_vocabulary(n_gram=1)
    de_character_vocabulary = load_de_character_vocabulary(n_gram=1)

    dfs = [split_to_tokens(df) for df in dfs]
    dfs = [parse_token_id(df, en_vocabulary, de_vocabulary) for df in dfs]
    dfs = [parse_character_id(df, en_character_vocabulary, de_character_vocabulary) for df in dfs]

    dfs = [df[df['en_tokens_length'] < max_sentence_length] for df in dfs]
    dfs = [df[df['de_tokens_length'] < max_sentence_length] for df in dfs]
    for i in range(len(dfs)):
        print(len(dfs[i]))

    if is_debug:
        dfs = [df[:100] for df in dfs]

    return dfs


def split_to_tokens(df):
    split_fn = lambda x: x.split()
    df['en_tokens'] = df['en_sentences'].map(split_fn)
    df['de_tokens'] = df['de_sentences'].map(split_fn)
    return df


def parse_token_id(df, en_vocabulary, de_vocabulary):
    df['en_tokens_ids'] = en_vocabulary.parse_text_without_pad(df['en_tokens'])
    df['de_tokens_ids'] = de_vocabulary.parse_text_without_pad(df['de_tokens'])
    df['en_tokens_length'] = df['en_tokens_ids'].map(len)
    df['de_tokens_length'] = df['de_tokens_ids'].map(len)
    return df


def parse_character_id(df, en_character_vocabulary, de_character_vocabulary):
    df['en_character_ids'] = en_character_vocabulary.parse_string_without_padding(df['en_tokens'])
    df['de_character_ids'] = de_character_vocabulary.parse_string_without_padding(df['de_tokens'])
    return df


if __name__ == '__main__':
    en_vocabulary = load_en_vocabulary()
    de_vocabulary = load_de_vocabulary()
    en_unk_id = en_vocabulary.word_to_id(en_vocabulary.unk)
    de_unk_id = de_vocabulary.word_to_id(de_vocabulary.unk)
    dfs = load_wmt14_en_de_data(is_debug=False)
    print('en_unk_id: {}, de_unk_id: {}'.format(en_unk_id, de_unk_id))
    print(dfs[0].iloc[0]['en_tokens'])
    print(dfs[0].iloc[0]['en_tokens_ids'])

    def count_df_unk(df):
        df['en_unk_count'] = df['en_tokens_ids'].map(count_en_unk)
        df['de_unk_count'] = df['de_tokens_ids'].map(count_de_unk)
        return df

    def count_en_unk(tokens):
        unk_count = [1 if tok == en_unk_id else 0 for tok in tokens]
        return sum(unk_count)

    def count_de_unk(tokens):
        unk_count = [1 if tok == de_unk_id else 0 for tok in tokens]
        return sum(unk_count)

    def print_unk_count(df):
        total_en_unk = sum(df['en_unk_count'])
        total_de_unk = sum(df['de_unk_count'])
        total_en_count = sum(df['en_tokens_length'])
        total_de_count = sum(df['de_tokens_length'])
        print('en unk count {}/{}, frac: {}'.format(total_en_unk, total_en_count, total_en_unk / total_en_count))
        print('de unk count {}/{}, frac: {}'.format(total_de_unk, total_de_count, total_de_unk / total_de_count))

    dfs = [count_df_unk(df) for df in dfs]
    train_df, valid_df, test_df = dfs
    print('train_df: {}, valid_df: {}, test_df: {}'.format(len(train_df), len(valid_df), len(test_df)))
    for df in dfs:
        print_unk_count(df)


