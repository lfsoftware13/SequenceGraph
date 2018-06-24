from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from read_data.method_naming.load_vocabulary import load_summarization_java_code_vocabulary, load_summarization_method_name_vocabulary
from read_data.method_naming.read_summarization_source_code_to_method_name_data import read_summarization_test_data, read_summarization_train_data_with_valid


@disk_cache(basename='read_summarization_data_with_id', directory=CACHE_DATA_PATH)
def read_summarization_data_with_id():
    java_code_vocab = load_summarization_java_code_vocabulary()
    method_name_vocab = load_summarization_method_name_vocabulary()
    train_df, valid_df = read_summarization_train_data_with_valid()
    test_df = read_summarization_test_data()
    # train_df = filter_output_long(train_df, max_len=10)
    # valid_df = filter_output_long(valid_df, max_len=10)
    # test_df = filter_output_long(test_df, max_len=10)

    parsed_train_df = parse_java_token_to_id(train_df, java_code_vocab, method_name_vocab)
    parsed_valid_df = parse_java_token_to_id(valid_df, java_code_vocab, method_name_vocab)
    parsed_test_df = parse_java_token_to_id(test_df, java_code_vocab, method_name_vocab)
    return parsed_train_df, parsed_valid_df, parsed_test_df


@disk_cache(basename='read_summarization_data_with_subtokens_list', directory=CACHE_DATA_PATH)
def read_summarization_data_with_subtokens_list():
    java_code_vocab = load_summarization_java_code_vocabulary()
    method_name_vocab = load_summarization_method_name_vocabulary()
    train_df, valid_df = read_summarization_train_data_with_valid()
    test_df = read_summarization_test_data()
    # train_df = filter_output_long(train_df, max_len=10)
    # valid_df = filter_output_long(valid_df, max_len=10)
    # test_df = filter_output_long(test_df, max_len=10)

    subtokens_train_df = transform_token_to_subtoken_list(train_df)
    subtokens_valid_df = transform_token_to_subtoken_list(valid_df)
    subtokens_test_df = transform_token_to_subtoken_list(test_df)
    parsed_train_df = parse_subtoken_java_token_to_id(subtokens_train_df, java_code_vocab, method_name_vocab)
    parsed_valid_df = parse_subtoken_java_token_to_id(subtokens_valid_df, java_code_vocab, method_name_vocab)
    parsed_test_df = parse_subtoken_java_token_to_id(subtokens_test_df, java_code_vocab, method_name_vocab)
    return parsed_train_df, parsed_valid_df, parsed_test_df


@disk_cache(basename='read_summarization_data_with_id_with_debug', directory=CACHE_DATA_PATH)
def read_summarization_data_with_id_with_debug(is_debug=False):
    parsed_train_df, parsed_valid_df, parsed_test_df = read_summarization_data_with_id()
    if is_debug:
        parsed_train_df, parsed_valid_df, parsed_test_df = parsed_train_df[:100], parsed_valid_df[:100], parsed_test_df[:100]
    return parsed_train_df, parsed_valid_df, parsed_test_df


@disk_cache(basename='read_summarization_data_with_subtokens_list_with_debug', directory=CACHE_DATA_PATH)
def read_summarization_data_with_subtokens_list_with_debug(is_debug=False):
    parsed_train_df, parsed_valid_df, parsed_test_df = read_summarization_data_with_subtokens_list()
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


if __name__ == '__main__':
    train_df, valid_df, test_df = read_summarization_data_with_subtokens_list_with_debug(False)
    print(train_df)
    print(valid_df)
    print(test_df.iloc[0])
    print(len(train_df), len(valid_df), len(test_df))



