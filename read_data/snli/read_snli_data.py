import json
import os
import pandas as pd

from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from config import snli_data_path


def read_jsonl_data_to_df(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
        objs = [json.loads(line.strip()) for line in lines]
    df = pd.DataFrame.from_records(objs)
    return df


@disk_cache(basename='snli.read_snli_train_data', directory=CACHE_DATA_PATH)
def read_snli_train_data():
    path = os.path.join(snli_data_path, 'snli_1.0_train.jsonl')
    df = read_jsonl_data_to_df(path)
    return df


@disk_cache(basename='snli.read_snli_valid_data', directory=CACHE_DATA_PATH)
def read_snli_valid_data():
    path = os.path.join(snli_data_path, 'snli_1.0_dev.jsonl')
    df = read_jsonl_data_to_df(path)
    return df


@disk_cache(basename='snli.read_snli_test_data', directory=CACHE_DATA_PATH)
def read_snli_test_data():
    test_path = os.path.join(snli_data_path, 'snli_1.0_test.jsonl')
    test_df = read_jsonl_data_to_df(test_path)
    return test_df


@disk_cache(basename='snli.read_snli_split_train_data', directory=CACHE_DATA_PATH)
def read_snli_split_train_data():
    train_df = read_snli_train_data()
    train_df = split_sentence_blank(train_df)
    return train_df


@disk_cache(basename='snli.read_snli_split_valid_data', directory=CACHE_DATA_PATH)
def read_snli_split_valid_data():
    valid_df = read_snli_valid_data()
    valid_df = split_sentence_blank(valid_df)
    return valid_df


@disk_cache(basename='snli.read_snli_split_test_data', directory=CACHE_DATA_PATH)
def read_snli_split_test_data():
    df = read_snli_test_data()
    df = split_sentence_blank(df)
    return df


def split_sentence_blank(df):
    split_fn = lambda x: x.strip().split()
    df['tokens1'] = df['sentence1'].map(split_fn)
    df['tokens2'] = df['sentence2'].map(split_fn)
    return df


if __name__ == '__main__':
    train_df = read_snli_split_train_data()
    valid_df = read_snli_split_valid_data()
    test_df = read_snli_split_test_data()
    def print_total_len(df):
        df['sentence1_len'] = df['tokens1'].map(len)
        df['sentence2_len'] = df['tokens2'].map(len)
        df['total_len'] = df['sentence1_len'] + df['sentence2_len']
        print(max(df['total_len']))
        print(len(df))
        df = df[df['total_len'] < 80]
        print(len(df))

    print_total_len(train_df)
    print_total_len(valid_df)
    print_total_len(test_df)
    # print(len(train_df))
    # print(len(valid_df))
    # print(len(matched_df))
    # print(len(mismatch_df))
    # for i in range(10):
    #     print('in {}'.format(i))
    #     print(matched_df.iloc[i]['sentence1'])
    #     print(matched_df.iloc[i]['sentence1_parse'])
    #     print(matched_df.iloc[i]['sentence1_binary_parse'])

