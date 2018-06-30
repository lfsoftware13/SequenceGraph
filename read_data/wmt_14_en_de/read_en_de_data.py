import os
import pandas as pd

from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from config import wmt2014_en_de_path


def read_sentence_in_lines(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip().lower() for line in lines]
    return lines


@disk_cache(basename='read_en_sentence_data', directory=CACHE_DATA_PATH)
def read_en_sentence_data():
    file_path = os.path.join(wmt2014_en_de_path,  'train.en')
    lines = read_sentence_in_lines(file_path)
    return lines


@disk_cache(basename='read_de_sentence_data', directory=CACHE_DATA_PATH)
def read_de_sentence_data():
    file_path = os.path.join(wmt2014_en_de_path, 'train.de')
    lines = read_sentence_in_lines(file_path)
    return lines


@disk_cache(basename='read_en_test_data', directory=CACHE_DATA_PATH)
def read_en_test_data():
    file_path = os.path.join(wmt2014_en_de_path, 'newstest2014.en')
    lines = read_sentence_in_lines(file_path)
    return lines


@disk_cache(basename='read_de_test_data', directory=CACHE_DATA_PATH)
def read_de_test_data():
    file_path = os.path.join(wmt2014_en_de_path, 'newstest2014.de')
    lines = read_sentence_in_lines(file_path)
    return lines


@disk_cache(basename='read_en_de_sentence_all_train_data', directory=CACHE_DATA_PATH)
def read_en_de_sentence_all_train_data():
    en_sentences = read_en_sentence_data()
    de_sentences = read_de_sentence_data()
    print('en sentence count: {}, de sentence count: {}'.format(len(en_sentences), len(de_sentences)))
    df = consist_sentence_to_df(en_sentences, de_sentences)
    return df


@disk_cache(basename='read_en_de_sentence_train_valid_data', directory=CACHE_DATA_PATH)
def read_en_de_sentence_train_valid_data():
    df = read_en_de_sentence_all_train_data()
    print('total train size: {}'.format(len(df)))
    valid_df = df.sample(frac=0.15)
    train_df = df.drop(valid_df.index)
    print('train size: {}, valid size: {}'.format(len(train_df), len(valid_df)))
    return train_df, valid_df


@disk_cache(basename='read_en_de_sentence_test_data', directory=CACHE_DATA_PATH)
def read_en_de_sentence_test_data():
    en_sentences = read_en_test_data()
    de_sentences = read_de_test_data()
    print('en test sentence count: {}, de test sentence count: {}'.format(len(en_sentences), len(de_sentences)))
    df = consist_sentence_to_df(en_sentences, de_sentences)
    return df


def consist_sentence_to_df(en_sentences, de_sentences):
    key_list = ['en_sentences', 'de_sentences']
    value_list = [en_sentences, de_sentences]
    data_dict = {key: value for key, value in zip(key_list, value_list)}
    df = pd.DataFrame.from_dict(data_dict)
    return df


if __name__ == '__main__':
    df = read_en_de_sentence_all_train_data()
    print('max en token length: ', max(df['en_length']))
    print('max de token length: ', max(df['de_length']))
    less_en_df = df[df['en_length'] < 50]
    less_de_df = df[df['de_length'] < 50]
    print('less than 50 en sentence count: {}, less than 50 de sentence count: {}'.format(len(less_en_df), len(less_de_df)))