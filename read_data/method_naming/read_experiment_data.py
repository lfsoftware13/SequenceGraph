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

    parsed_train_df = parse_java_token_to_id(train_df, java_code_vocab, method_name_vocab)
    parsed_valid_df = parse_java_token_to_id(valid_df, java_code_vocab, method_name_vocab)
    parsed_test_df = parse_java_token_to_id(test_df, java_code_vocab, method_name_vocab)
    return parsed_train_df, parsed_valid_df, parsed_test_df


@disk_cache(basename='read_summarization_data_with_id_with_debug', directory=CACHE_DATA_PATH)
def read_summarization_data_with_id_with_debug(is_debug=False):
    parsed_train_df, parsed_valid_df, parsed_test_df = read_summarization_data_with_id()
    if is_debug:
        parsed_train_df, parsed_valid_df, parsed_test_df = parsed_train_df[:100], parsed_valid_df[:100], parsed_test_df[:100]
    return parsed_train_df, parsed_valid_df, parsed_test_df


def parse_java_token_to_id(df, java_code_vocab, method_name_vocab):
    parse_tokens_fn = lambda text: java_code_vocab.parse_text_without_pad([text])[0]
    df['tokens_id'] = df['tokens'].map(parse_tokens_fn)
    parse_names_fn = lambda name: method_name_vocab.parse_text_without_pad([name], use_position_label=True)[0]
    df['names_id'] = df['name'].map(parse_names_fn)
    return df


if __name__ == '__main__':
    train_df, valid_df, test_df = read_summarization_data_with_id_with_debug(False)
    print(train_df)
    print(valid_df)
    print(test_df.iloc[0])
    print(len(train_df), len(valid_df), len(test_df))



