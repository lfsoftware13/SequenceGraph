from common.constants import CACHE_DATA_PATH
from common.util import disk_cache, PaddedList
from read_data.data_set import OriDataSet
from read_data.method_naming.load_vocabulary import load_summarization_java_code_vocabulary, load_summarization_method_name_vocabulary
from read_data.method_naming.read_summarization_source_code_to_method_name_data import read_summarization_test_data, read_summarization_train_data_with_valid

import pandas as pd


@disk_cache(basename='read_summarization_data_with_id', directory=CACHE_DATA_PATH)
def read_summarization_data_with_id():
    java_code_vocab = load_summarization_java_code_vocabulary()
    method_name_vocab = load_summarization_method_name_vocabulary()
    train_df, valid_df = read_summarization_train_data_with_valid()
    test_df = read_summarization_test_data()
    train_df = filter_output_long(train_df, max_len=10)
    valid_df = filter_output_long(valid_df, max_len=10)
    test_df = filter_output_long(test_df, max_len=10)

    parsed_train_df = parse_java_token_to_id(train_df, java_code_vocab, method_name_vocab)
    parsed_valid_df = parse_java_token_to_id(valid_df, java_code_vocab, method_name_vocab)
    parsed_test_df = parse_java_token_to_id(test_df, java_code_vocab, method_name_vocab)
    return parsed_train_df, parsed_valid_df, parsed_test_df


@disk_cache(basename='read_summarization_data_with_subtokens_list', directory=CACHE_DATA_PATH)
def read_summarization_data_with_subtokens_list(max_token_length):
    java_code_vocab = load_summarization_java_code_vocabulary()
    method_name_vocab = load_summarization_method_name_vocabulary()
    train_df, valid_df = read_summarization_train_data_with_valid()
    test_df = read_summarization_test_data()
    train_df = filter_output_long(train_df, max_len=10)
    valid_df = filter_output_long(valid_df, max_len=10)
    test_df = filter_output_long(test_df, max_len=10)

    subtokens_train_df = transform_token_to_subtoken_list(train_df)
    subtokens_valid_df = transform_token_to_subtoken_list(valid_df)
    subtokens_test_df = transform_token_to_subtoken_list(test_df)

    filter_train_df = filter_input_long(subtokens_train_df, max_len=max_token_length)
    filter_valid_df = filter_input_long(subtokens_valid_df, max_len=max_token_length)
    filter_test_df = filter_input_long(subtokens_test_df, max_len=max_token_length)

    parsed_train_df = parse_subtoken_java_token_to_id(filter_train_df, java_code_vocab, method_name_vocab)
    parsed_valid_df = parse_subtoken_java_token_to_id(filter_valid_df, java_code_vocab, method_name_vocab)
    parsed_test_df = parse_subtoken_java_token_to_id(filter_test_df, java_code_vocab, method_name_vocab)
    return parsed_train_df, parsed_valid_df, parsed_test_df


@disk_cache(basename='read_summarization_data_with_id_with_debug', directory=CACHE_DATA_PATH)
def read_summarization_data_with_id_with_debug(is_debug=False):
    parsed_train_df, parsed_valid_df, parsed_test_df = read_summarization_data_with_id()
    if is_debug:
        parsed_train_df, parsed_valid_df, parsed_test_df = parsed_train_df[:100], parsed_valid_df[:100], parsed_test_df[:100]
    return parsed_train_df, parsed_valid_df, parsed_test_df


@disk_cache(basename='read_summarization_data_with_subtokens_list_with_debug', directory=CACHE_DATA_PATH)
def read_summarization_data_with_subtokens_list_with_debug(is_debug=False, max_token_length=200):
    parsed_train_df, parsed_valid_df, parsed_test_df = read_summarization_data_with_subtokens_list(max_token_length)
    if is_debug:
        parsed_train_df, parsed_valid_df, parsed_test_df = parsed_train_df[:100], parsed_valid_df[:100], parsed_test_df[:100]
    return parsed_train_df, parsed_valid_df, parsed_test_df


def parse_java_token_to_id(df, java_code_vocab, method_name_vocab):
    parse_tokens_fn = lambda text: java_code_vocab.parse_text_without_pad([text])[0]
    df['tokens_id'] = df['tokens'].map(parse_tokens_fn)
    parse_names_fn = lambda name: method_name_vocab.parse_text_without_pad([name], use_position_label=True)[0]
    df['names_id'] = df['name'].map(parse_names_fn)
    return df


def parse_subtoken_java_token_to_id(df, java_code_vocab, method_name_vocab):
    parse_tokens_fn = lambda text: java_code_vocab.parse_text_without_pad(text)
    df['subtokens_id'] = df['subtokens'].map(parse_tokens_fn)
    parse_names_fn = lambda name: method_name_vocab.parse_text_without_pad([name], use_position_label=True)[0]
    df['names_id'] = df['name'].map(parse_names_fn)
    return df


def transform_token_to_subtoken_list(df):
    id_start = '<id>'
    id_end = '</id>'
    count = 0
    def transform_token_subtoken(tokens):
        nonlocal count
        count += 1
        if count % 1000 == 0:
            print(count)
        in_id = False
        subtoken_list = []
        one_subtoken = []

        for tok in tokens:
            if not in_id:
                if tok == id_start:
                    in_id = True
                    one_subtoken = []
                    continue
                elif tok == id_end:
                    # print('error id end: {}'.format(tokens))
                    print('error id end: {}')
                    return None
                subtoken_list += [[tok]]
            else:
                if tok == id_start:
                    # print('nesting id: {}'.format(tokens))
                    print('nesting id: {}')
                    return None
                if tok == id_end:
                    in_id = False
                    subtoken_list += [one_subtoken]
                    continue
                one_subtoken += [tok]
        return subtoken_list
    df['subtokens'] = df['tokens'].map(transform_token_subtoken)
    print(len(df))
    df = df[df['subtokens'].map(lambda x: x is not None)]
    print(len(df))
    return df


def filter_output_long(df, max_len=10):
    length = df['name'].map(len)
    select = length < max_len
    df = df[select]
    return df


@disk_cache(basename='load_method_naming_data', directory=CACHE_DATA_PATH)
def load_method_naming_data(decoder_max_length, is_debug=False, ):
    from read_data.method_naming.load_vocabulary import load_summarization_java_code_vocabulary, \
        load_summarization_method_name_vocabulary
    code_vocabulary = load_summarization_java_code_vocabulary()
    method_name_vocabulary = load_summarization_method_name_vocabulary()
    embedding_size = code_vocabulary.vocabulary_size
    pad_id = code_vocabulary.word_to_id(code_vocabulary.pad)
    unk_id = code_vocabulary.word_to_id(code_vocabulary.unk)
    begin_id = code_vocabulary.word_to_id(code_vocabulary.begin_tokens[0])
    hole_id = code_vocabulary.word_to_id(code_vocabulary.hole_token)
    output_pad_id = method_name_vocabulary.word_to_id(method_name_vocabulary.pad)
    output_size = method_name_vocabulary.vocabulary_size

    def parse_data(df: pd.DataFrame):
        res = []
        for row in df.iterrows():
            row = row[1]
            res.append(
                {
                    "text": row['tokens'],
                    "input": row['subtokens_id'],
                    "label": PaddedList(row['names_id'],
                                        fill_value=output_pad_id,
                                        shape=[decoder_max_length, ]),
                    "length": len(row['subtokens_id']),
                    "max_decoder_length": decoder_max_length,
                }
            )
        return res

    train, valid, test = [OriDataSet(parse_data(t))
                          for t in
                          read_summarization_data_with_subtokens_list_with_debug(is_debug)]

    return train, valid, test, embedding_size, pad_id, unk_id, begin_id, hole_id, output_size, output_pad_id


def filter_input_long(df, max_len=300):
    length = df['subtokens'].map(len)
    select = length < max_len
    df = df[select]
    return df


def filter_key_length(df, key, max_len):
    length = df[key].map(len)
    select = length < max_len
    df = df[select]
    return df



if __name__ == '__main__':
    train_df, valid_df, test_df = read_summarization_data_with_subtokens_list_with_debug(False, max_token_length=200)
    train_len = train_df['subtokens'].map(len)
    valid_len = valid_df['subtokens'].map(len)
    test_len = test_df['subtokens'].map(len)
    # print(train_df.iloc[0])
    print(len(train_df))
    print(len(valid_df))
    print(len(test_df))
    # print(max(train_len))
    # out = train_df[train_len > 200]
    less_train = train_df[train_len < 200]
    less_valid = valid_df[valid_len < 200]
    less_test = test_df[test_len < 200]
    # print(out['subtokens'].iloc[0])
    print(len(less_train))
    print(len(less_valid))
    print(len(less_test))



