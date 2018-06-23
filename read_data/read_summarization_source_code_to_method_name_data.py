# The dataset used for the paper experiments contains the following folders:
#
# train: Contains all the training (.java) files
#
# test: Contains all the test (.java) files
#
# json: Contains a parsed form of the data, that can be easily input into
#   machine learning models. The format of the json files is explained below.
#
#
#
# ========================== JSON file format ==========================
# Each .json file is a list of methods.
#
# Each method is described by a dictionary that contains the following key-value
# pairs:
#   filename: the origin of the method
#   name: a list of the normalized subtokens of the method name
#   tokens: a list of the tokens of the code within the body of the method. The
#     code tokens are padded with a special <SENTENCE_START> and <SENTENCE_END>
#     symbol. Source code identifiers (ie. variable, method and type names) are
#     annotated by surrounding them with `<id>` and `</id>` tags. These tags
#     were removed as a preprocessing step in this paper.
import json
import os

import pandas as pd

from common.constants import CACHE_DATA_PATH
from common.util import iterate_directory, disk_cache
from config import summarization_source_code_to_method_name_path


def read_summarization_json_file(json_path: str):
    """
    read json file and parse content as pandas.DataFrame
    :param path: json file path
    :return:
    """
    df = pd.read_json(json_path, encoding='utf-8')
    return df


@disk_cache(basename='read_summarization_train_data', directory=CACHE_DATA_PATH)
def read_summarization_train_data():
    parent_path = summarization_source_code_to_method_name_path
    all_train_file = r'all_train_methodnaming.json'
    total_train_path = os.path.join(parent_path, all_train_file)
    df = read_summarization_json_file(total_train_path)
    return df


@disk_cache(basename='read_summarization_train_data_with_valid', directory=CACHE_DATA_PATH)
def read_summarization_train_data_with_valid():
    train_df = read_summarization_train_data()
    # train_df = train_df[:100]
    print(len(train_df))
    train_df, valid_df = split_valid_data(train_df)
    print(len(train_df), len(valid_df))
    return train_df, valid_df


@disk_cache(basename='read_summarization_test_data_list', directory=CACHE_DATA_PATH)
def read_summarization_test_data_list():

    def filter_special_summarization_file_name(filename, data_type=None, shuffled=False):
        file_name, extension_name = os.path.splitext(filename)
        file_subnames = file_name.split('_')
        if data_type is not None and file_subnames[1] != data_type:
            return False
        if shuffled and (len(file_subnames) < 4 or file_subnames[3] != 'shuffled'):
            return False
        return True

    parent_path = summarization_source_code_to_method_name_path
    df_list = []
    for full_file, file in iterate_directory(parent_path, extensions=['.json']):
        if not filter_special_summarization_file_name(file, data_type='test', shuffled=False):
            continue
        df = read_summarization_json_file(full_file)
        df_list += [df]
    return df_list


@disk_cache(basename='read_summarization_test_data', directory=CACHE_DATA_PATH)
def read_summarization_test_data():
    df_list = read_summarization_test_data_list()
    df = pd.concat(df_list, ignore_index=False)
    return df


@disk_cache(basename='read_summarization_shuffled_train_data', directory=CACHE_DATA_PATH)
def read_summarization_shuffled_train_data():
    parent_path = summarization_source_code_to_method_name_path
    train_file = r'libgdx_train_methodnaming_shuffled.json'
    train_path = os.path.join(parent_path, train_file)
    df = read_summarization_json_file(train_path)
    return df


@disk_cache(basename='read_summarization_shuffled_test_data', directory=CACHE_DATA_PATH)
def read_summarization_shuffled_test_data():
    parent_path = summarization_source_code_to_method_name_path
    test_file = r'libgdx_test_methodnaming_shuffled.json'
    test_path = os.path.join(parent_path, test_file)
    df = read_summarization_json_file(test_path)
    return df


def split_valid_data(df: pd.DataFrame):
    count = 0

    def not_select_fn(x):
        nonlocal count
        if count % 100 == 0:
            print(count)
        count += 1
        res = x not in select_filename.values
        return res

    print('before split')
    valid_df= df.sample(frac=0.2)
    select_filename = valid_df['filename']
    print(len(select_filename))
    not_valid = df['filename'].map(not_select_fn)
    train_df = df[not_valid]

    return train_df, valid_df


if __name__ == '__main__':
    json_path = r'G:\Project\dataset\json\gradle_test_methodnaming.json'
    # df = read_summarization_json_file(json_path)
    train, valid = read_summarization_train_data_with_valid()



