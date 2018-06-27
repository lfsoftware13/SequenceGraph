import json
import os
import pandas as pd

from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from config import multinli_data_path


def read_jsonl_data_to_df(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        objs = [json.loads(line) for line in lines]
    df = pd.DataFrame.from_records(objs)
    return df


def read_multinli_train_data():
    path = os.path.join(multinli_data_path, 'multinli_1.0_train.jsonl')
    df = read_jsonl_data_to_df(path)
    return df


@disk_cache(basename='read_multinli_test_data', directory=CACHE_DATA_PATH)
def read_multinli_test_data():
    matched_path = os.path.join(multinli_data_path, 'multinli_1.0_dev_matched.jsonl')
    mismatched_path = os.path.join(multinli_data_path, 'multinli_1.0_dev_mismatched.jsonl')
    matched_df = read_jsonl_data_to_df(matched_path)
    mismatched_df = read_jsonl_data_to_df(mismatched_path)
    return matched_df, mismatched_df


@disk_cache(basename='read_multinli_train_valid_data', directory=CACHE_DATA_PATH)
def read_multinli_train_valid_data():
    df = read_multinli_train_data()
    valid_df = df.sample(frac=0.15)
    train_df = df.drop(valid_df.index)
    return train_df, valid_df


@disk_cache(basename='read_multinli_split_test_data', directory=CACHE_DATA_PATH)
def read_multinli_split_test_data():
    matched_df, mismatched_df = read_multinli_test_data()
    matched_df = split_sentence_blank(matched_df)
    mismatched_df = split_sentence_blank(mismatched_df)
    return matched_df, mismatched_df


@disk_cache(basename='read_multinli_split_train_and_valid_data', directory=CACHE_DATA_PATH)
def read_multinli_split_train_and_valid_data():
    train_df, valid_df = read_multinli_train_valid_data()
    train_df = split_sentence_blank(train_df)
    valid_df = split_sentence_blank(valid_df)
    return train_df, valid_df


def split_sentence_blank(df):
    split_fn = lambda x: x.split(' ')
    df['tokens1'] = df['sentence1'].map(split_fn)
    df['tokens2'] = df['sentence2'].map(split_fn)
    return df


if __name__ == '__main__':
    train_df, valid_df = read_multinli_train_valid_data()
    matched_df, mismatch_df = read_multinli_test_data()
    def print_total_len(df):
        split_len_fn = lambda x: len(x.split(' '))
        df['sentence1_len'] = df['sentence1'].map(split_len_fn)
        df['sentence2_len'] = df['sentence2'].map(split_len_fn)
        df['total_len'] = df['sentence1_len'] + df['sentence2_len']
        print(max(df['total_len']))
        print(len(df))
        df = df[df['total_len'] < 80]
        print(len(df))

    print_total_len(train_df)
    print_total_len(valid_df)
    print_total_len(matched_df)
    print_total_len(mismatch_df)
    # print(len(train_df))
    # print(len(valid_df))
    # print(len(matched_df))
    # print(len(mismatch_df))
    # for i in range(10):
    #     print('in {}'.format(i))
    #     print(matched_df.iloc[i]['sentence1'])
    #     print(matched_df.iloc[i]['sentence1_parse'])
    #     print(matched_df.iloc[i]['sentence1_binary_parse'])

