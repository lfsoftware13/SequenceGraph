import random
import more_itertools
import pandas as pd
import typing
from toolz.sandbox import unzip
from torch.utils.data import Dataset

from common import util
from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from read_data.data_set import OriDataSet


def transform1(seq):
    return [t for t in seq]


def transform2(seq):
    return [t for t in reversed(seq)]


def transform3(seq):
    return list(more_itertools.flatten([t if len(t) < 2 else [t[1], t[0]] for t in more_itertools.chunked(seq, 2)]))


def generate_one_seq(max_index=10, max_length=10, min_length=5):
    length = random.randint(min_length, max_length)
    return [random.randint(0, max_index) for _ in range(length)]


def create_target(seq):
    i = random.randint(0, 2)
    return [transform1, transform2, transform3][i](seq), i


def generate_data(max_index=10, max_length=10, min_length=5, number=100000):
    df = pd.DataFrame()
    df['x'] = [generate_one_seq(max_index, max_length, min_length) for _ in range(number)]
    target = [create_target(t) for t in df['x']]
    y, transform_id = [list(t) for t in unzip(target)]
    df['y'] = y
    df['transform_id'] = transform_id
    return df


@disk_cache(basename='sequence_transform_data.load_generated_data', directory=CACHE_DATA_PATH)
def load_generated_data(is_debug):
    import os
    import config
    df = pd.read_pickle(os.path.join(config.sequence_data_path, 'sequence_data.pkl'))

    def to_dict(df: pd.DataFrame):
        res = []
        for row in df.iterrows():
            row = row[1]
            res.append(
                {
                    'x': row.x,
                    'y': row.y,
                    'transform_id': row.transform_id
                }
            )
        return res

    test = df.sample(n=10000, )
    df = df.drop(test.index)
    valid = df.sample(n=10000, )
    train = df.drop(valid.index)

    if is_debug:
        train, valid, test = [t.head(100) for t in [train, valid, test]]

    return [OriDataSet(to_dict(t)) for t in [train, valid, test]]


def generate_source_data(max_index=10, max_length=10, min_length=5, number=100000):
    df = pd.DataFrame()
    df['x'] = [generate_one_seq(max_index, max_length, min_length) for _ in range(number)]
    return df


class RandomTargetDataSet(Dataset):
    def __init__(self, source_data: typing.List[typing.List]):
        super().__init__()
        self._source_data = source_data

    def __getitem__(self, index):
        x = self._source_data[index]
        y = create_target(x)[0]
        return {'x': x, 'y': y}

    def __len__(self):
        return len(self._source_data)


@disk_cache(basename='sequence_transform_data.load_generated_random_target_data', directory=CACHE_DATA_PATH)
def load_generated_random_target_data(is_debug):
    import os
    import config
    df = pd.read_pickle(os.path.join(config.sequence_data_path, 'source_sequence_data.pkl'))

    def to_dict(df: pd.DataFrame):
        res = []
        for row in df.iterrows():
            row = row[1]
            res.append(row.x)
        return res
    test = df.sample(n=10000, )
    df = df.drop(test.index)
    valid = df.sample(n=10000, )
    train = df.drop(valid.index)

    if is_debug:
        train, valid, test = [t.head(100) for t in [train, valid, test]]

    train, valid, test = [to_dict(t) for t in [train, valid, test]]

    valid = list(filter(lambda x: x not in train, valid))
    test = list(filter(lambda x: x not in train, test))
    test = list(filter(lambda x: x not in valid, test))

    return [RandomTargetDataSet(t) for t in [train, valid, test]]


if __name__ == '__main__':
    import config
    import os
    s = generate_one_seq()
    print("random select one sequence:{}".format(s))
    print("transform1 on the seq:{}".format(transform1(s)))
    print("transform2 on the seq:{}".format(transform2(s)))
    print("transform3 on the seq:{}".format(transform3(s)))
    for _ in range(10):
        print("create target on the sequence:{}".format(create_target(s)))
    print(generate_data(10, 10, 5, 10))
    # df = generate_data(10, 20, 10, )
    df = generate_source_data(10, 20, 10, )
    util.make_dir(config.sequence_data_path)
    df.to_pickle(os.path.join(config.sequence_data_path, 'source_sequence_data.pkl'))
    # train, valid, test = load_generated_data()
