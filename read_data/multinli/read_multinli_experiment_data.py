from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from read_data.multinli.load_multinli_vocabulary import load_multinli_vocabulary
from read_data.multinli.read_multinli_data import read_multinli_train_valid_data, read_multinli_test_data, \
    read_multinli_split_train_and_valid_data, read_multinli_split_test_data


@disk_cache(basename='load_multinli_data', directory=CACHE_DATA_PATH)
def load_multinli_data(is_debug, max_total_len=80):
    """
    sentence1: sentence 1
    tokens1: token list of sentence 1, split by ' '
    tokens_id1: token id list of sentence 1 with begin token 1 and end token 1
    sentence2: like before
    tokens2:  like before
    tokens_id2: like before
    tokens_len1: token 1 length
    tokens_len2:
    total_len: token_len1 + token_len2
    gold_label: target label. one of ['entailment', 'neutral', 'contradiction']
    label: target label id [0, 1, 2].
    :param is_debug:
    :param max_total_len:
    :return: dataframe obj
    """
    train_df, valid_df = read_multinli_split_train_and_valid_data()
    matched_df, mismatched_df = read_multinli_split_test_data()
    dfs = [train_df, valid_df, matched_df, mismatched_df]
    print('after read data: ')
    for i in range(len(dfs)):
        print(len(dfs[i]))
    vocab = load_multinli_vocabulary()
    dfs = [parse_tokens_id(df, vocab) for df in dfs]
    dfs = [parse_label_id(df) for df in dfs]
    print('after parse token to id')

    dfs = [df[df['total_len'] < max_total_len] for df in dfs]
    print('after filter longer than {} data: '.format(max_total_len))
    for i in range(len(dfs)):
        print(len(dfs[i]))

    if is_debug:
        dfs = [df[:100] for df in dfs]

    return dfs


def parse_tokens_id(df, vocab):
    df['tokens_id1'] = vocab.parse_text_without_pad(df['tokens1'], position_label_index=0)
    df['tokens_id2'] = vocab.parse_text_without_pad(df['tokens2'], position_label_index=1)
    df['tokens_len1'] = df['tokens_id1'].map(len)
    df['tokens_len2'] = df['tokens_id2'].map(len)
    df['total_len'] = df['tokens_len1'] + df['tokens_len2']
    return df


def parse_label_id(df):
    df = df[df['gold_label'] != '-']
    result_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    parse_fn = lambda x: result_dict[x]
    df['label'] = df['gold_label'].map(parse_fn)
    return df


if __name__ == '__main__':
    train_df, valid_df, matched_df, mismatched_df = load_multinli_data(is_debug=False)
    print(len(train_df))
    print(len(valid_df))
    print(len(matched_df))
    print(len(mismatched_df))
    train_df = train_df[train_df['gold_label'] != '-']
    valid_df = valid_df[valid_df['gold_label'] != '-']
    matched_df = matched_df[matched_df['gold_label'] != '-']
    mismatched_df = mismatched_df[mismatched_df['gold_label'] != '-']
    print(len(train_df))
    print(len(valid_df))
    print(len(matched_df))
    print(len(mismatched_df))

    for i in range(10):
        print(train_df.iloc[i]['label'])
