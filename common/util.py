import errno
import functools
import itertools
import os
import pickle
import random
import re
import types
from multiprocessing import Pool
import typing
import hashlib
import numpy as np

import copy
from typing import Iterator

import more_itertools
import sklearn
import pandas as pd
import sys
# import cytoolz as toolz
import toolz
import collections

import time

from sklearn.utils import shuffle


def make_dir(*path: str) -> None:
    """
    This method will recursively create the directory
    :param path: a variable length parameter
    :return:
    """
    path = os.path.join(*path)

    if not path:
        return

    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError("The path {} already exits but it is not a directory".format(path))
        return

    base, name = os.path.split(path)
    make_dir(base)
    if name:
        os.mkdir(path)


def format_dict_to_string(to_format_dict: dict) -> str:
    """
    :param to_format_dict: a dict to format
    :return:
    """

    def to_str(o):
        if is_sequence(o):
            return ''.join(to_str(t) for t in o)
        else:
            return str(o)
    # print(len('__'.join(to_str(a)+to_str(b) for a, b in to_format_dict.items())))
    return '__'.join(to_str(a)+to_str(b) for a, b in to_format_dict.items())


def ensure_directory(directory):
    """
    Create the directories along the provided directory path that do not exist.
    """
    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def disk_cache(basename, directory, method=False):
    """
    Function decorator for caching pickleable return values on disk. Uses a
    hash computed from the function arguments for invalidation. If 'method',
    skip the first argument, usually being self or cls. The cache filepath is
    'directory/basename-hash.pickle'.
    """
    directory = os.path.expanduser(directory)
    ensure_directory(directory)

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            key = (tuple(args), tuple(kwargs.items()))
            # Don't use self or cls for the invalidation hash.
            if method and key:
                key = key[1:]
            filename = '{}-{}.pickle'.format(basename, data_hash(key))
            print("the cache name is {}".format(filename))
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                print("load file from:{}".format(filepath))
                with open(filepath, 'rb') as handle:
                    return pickle.load(handle)
            result = func(*args, **kwargs)
            with open(filepath, 'wb') as handle:
                print("write cache to: {}".format(filepath))
                pickle.dump(result, handle)
            return result
        return wrapped

    return wrapper


def data_hash(key):

    def hash_value(hash_item):
        v = 0
        try:
            v = int(hashlib.md5(str(hash_item).encode('utf-8')).hexdigest(), 16)
        except Exception as e:
            print('error occur while hash item {} '.format(type(hash_item)))
        return v

    hash_val = 0
    key = list(more_itertools.flatten(key))
    for item in key:
        if isinstance(item, pd.DataFrame):
            serlist = [item.itertuples(index=False, name=None)]
            serlist = list(more_itertools.collapse(serlist))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        elif isinstance(item, pd.Series):
            serlist = item.tolist()
            serlist = list(more_itertools.collapse(serlist))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        elif isinstance(item, int) or isinstance(item, float) or isinstance(item, str):
            val = hash_value(item)
            hash_val += val
        elif isinstance(item, list) or isinstance(item, set) or isinstance(item, tuple):
            serlist = list(more_itertools.collapse(item))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        elif isinstance(item, dict):
            serlist = list(more_itertools.collapse(item.items()))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        else:
            print('type {} cant be hashed.'.format(type(item)))
    return str(hash_val)

# ================================================================
# multiprocess function
# ================================================================

def parallel_map(core_num, f, args):
    """
    :param core_num: the cpu number
    :param f: the function to parallel to do
    :param args: the input args
    :return:
    """

    with Pool(core_num) as p:
        r = p.map(f, args)
        return r

# ================================================================
# dict function
# ================================================================

def reverse_dict(d: dict) -> dict:
    """
    swap key and value of a dict
    dict(key->value) => dict(value->key)
    """
    return dict(map(reversed, d.items()))

# ================================================================
# sequence function
# ================================================================

def is_sequence(s):
    try:
        iterator = iter(s)
    except TypeError:
        return False
    else:
        if isinstance(s, str):
            return False
        return True


def convert_to_list(s):
    if is_sequence(s):
        return list(s)
    else:
        return [s]


def sequence_sum(itr):
    return sum(itr)

def padded_code_new(batch_code, fill_value):
    if not isinstance(batch_code, list):
        return batch_code
    elif not isinstance(batch_code[0], list):
        return batch_code

    batch_root = batch_code
    while True:
        if not isinstance(batch_root, list):
            return batch_code
        elif not isinstance(batch_root[0], list):
            return batch_code
        cur_fill_value = fill_value
        if isinstance(batch_root[0][0], list):
            cur_fill_value = []
        max_len = max(map(len, batch_root))
        for b in batch_root:
            while len(b) < max_len:
                b.append(cur_fill_value)
        # list(map(lambda x: list(more_itertools.padded(x, fillvalue=fill_value, n=max_len)), batch_root))

        tmp = []
        for son in batch_root:
            for s in son:
                tmp.append(s)
        batch_root = tmp

def padded(x, deepcopy=False, fill_value=0):
    import copy
    if deepcopy:
        x = copy.deepcopy(x)
    if not isinstance(x, list):
        return x
    elif isinstance(x[0], list):
        return padded_code_new(x, fill_value=fill_value)
    else:
        return x

def padded_to_length(x, length, fill_value):
    res = list(more_itertools.padded(x, fill_value, length))
    return res


def batch_holder(*data: typing.List, batch_size=32,):
    """
    :param data:
    :return:
    """
    def iterator():
        def one_epoch():
            i_data = list(map(lambda x: more_itertools.chunked(x, batch_size), data))
            return zip(*i_data)
        for i ,m in enumerate(more_itertools.repeatfunc(one_epoch, times=1)):
            for t in m:
                yield t

    return iterator

def dataset_holder(*args):
    def f():
        return args
    return f

def train_test_split(data, test_size):
    from sklearn.model_selection import train_test_split
    data = train_test_split(*data, test_size=test_size)

    d_len = len(data)
    train_data = [data[i] for i in range(0, d_len, 2)]
    test_data = [data[i] for i in range(1, d_len, 2)]
    return train_data, test_data

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def get_count_size(x):
    r = 0
    if hasattr(x, '__iter__') and not isinstance(x, (str, bytes, bytearray)):
        r += sum(get_count_size(t) for t in x)
    else:
        r += 1

    return r


def unique_adjacent(seq: typing.Iterator):
    pre = next(seq)
    yield pre
    for token in seq:
        if token == pre:
            continue
        else:
            pre = token
            yield pre


def group_df_to_grouped_list(data_df, groupby_key):
    grouped = data_df.groupby(groupby_key)
    group_list = []
    for name, group in grouped:
        group_list += [group]
    return group_list


def filter_length(df, limit_length, tokenize_fn, code_key='similar_code'):
    df['tokens'] = df[code_key].map(tokenize_fn)
    df = df[df['tokens'].map(lambda x: len(x) < limit_length)].copy()
    df = df.drop(columns=['tokens'], axis=1)
    return df


def maintain_function_co_firstlineno(ori_fn):
    """
    This decorator is used to make the decorated function's co_firstlineno the same as the ori_fn
    """

    def wrapper(fn):
        wrapper_code = fn.__code__
        fn.__code__ = types.CodeType(
            wrapper_code.co_argcount,
            wrapper_code.co_kwonlyargcount,
            wrapper_code.co_nlocals,
            wrapper_code.co_stacksize,
            wrapper_code.co_flags,
            wrapper_code.co_code,
            wrapper_code.co_consts,
            wrapper_code.co_names,
            wrapper_code.co_varnames,
            wrapper_code.co_filename,
            wrapper_code.co_name,
            ori_fn.__code__.co_firstlineno,
            wrapper_code.co_lnotab,
            wrapper_code.co_freevars,
            wrapper_code.co_cellvars
        )

        return fn

    return wrapper


def show_process_map(fn, l, print_steps=1000, error_default_value=None):
    res = []
    begin_time = time.time()
    fail_number = 0
    for i, t in enumerate(l):
        if i % print_steps == 0:
            print("{}/{} finished".format(i, len(l)))
            print("{}/{} data map failed".format(fail_number, len(l)))
        try:
            res.append(fn(t))
        except Exception as e:
            # print(e)
            fail_number += 1
            res.append(error_default_value)
    print("This map use {} seconds".format(time.time()-begin_time))
    print("{}/{} data map failed".format(fail_number, len(l)))
    return res


def inplace_show_process_map(fn, l, print_steps=1000, error_default_value=None):
    begin_time = time.time()
    fail_number = 0
    for i, t in enumerate(l):
        if i % print_steps == 0:
            print("{}/{} finished".format(i, len(l)))
            print("{}/{} data map failed".format(fail_number, len(l)))
        try:
            res = fn(t)
        except Exception as e:
            # print(e)
            fail_number += 1
            res = error_default_value
        l[i] = res
    print("This map use {} seconds".format(time.time() - begin_time))
    print("{}/{} data map failed".format(fail_number, len(l)))
    return l


@toolz.curry
def generate_mask(mask_index, size):
    '''
    :param mask_index: a iterable container of index
    :param size: the max size
    :return: a 0-1 mask list with the size shape
    '''
    res = MaskList(size, 0)
    for i in mask_index:
        if isinstance(i, int):
            res.set_mask(i)
        elif isinstance(i, tuple):
            res.set_mask(i[0], i[1])
    res.sort()
    return res


def data_loader(dataset, batch_size, is_shuffle=True, drop_last=False, epoch_ratio=1.0):
    idxs = list(range(len(dataset)))
    if is_shuffle:
        idxs = shuffle(idxs)
    idxs = idxs[0: int(len(idxs)*epoch_ratio)]
    # print("the idxs length:{}".format(len(idxs)))
    for idx in batch_holder(idxs, batch_size=batch_size)():
        idx = idx[0]
        if drop_last and len(idx) != batch_size:
            # print("drop_last:{}".format(drop_last))
            # print("len(idx) != batch_size: {}".format(len(idx) != batch_size))
            # print("to break")
            break
        batch = [dataset[i] for i in idx]
        # print("before yield")
        yield toolz.merge_with(lambda x:x, batch)


# ---------------------------------- PaddedList ------------------------------------------- #

class PaddedList(collections.Sequence):
    """
    list() -> new empty list
    list(iterable) -> new list initialized from iterable's items
    """

    def __init__(self, l, fill_value=0, shape=None):
        self.l = l
        self.fill_value = fill_value

        self.shape = self._l_shape(self.l) if shape is None else shape


    def _l_shape(self, l):
        if not isinstance(l, collections.Sized) and not isinstance(l, collections.Iterable):
            return []
        sh = [len(l)]

        cur_max_shape = None
        for one in l:
            one_shape = self._l_shape(one)
            cur_max_shape = self._cal_max_shapes(cur_max_shape, one_shape) if cur_max_shape is not None else one_shape

        if cur_max_shape is not None:
            sh += cur_max_shape
        return sh

    def _cal_max_shapes(self, ori_shape, one_shape):
        if len(ori_shape) != len(one_shape):
            raise ShapeDifferentException('Shape error. There are different shape in list. original shape is {}, current shape is {}'.format(ori_shape, one_shape))

        max_shape = []
        for ori, one in zip(ori_shape, one_shape):
            max_shape += [max(ori, one)]
        return max_shape

    # make sure the len(l_shape) == len(shape). This example l = [1, 2, 3], shape = [4, 4] will not appear.
    # the fill list and fill number will always append to the end
    def _create_list_as_shape(self, l, shape, fill_value=0):
        if not isinstance(l, collections.Sized) and not isinstance(l, collections.Iterable):
            if len(shape) > 0:
                raise ListShapeErrorException('the depth of list is smaller than len(shape).')
        if len(shape) <= 0:
            raise ListShapeErrorException('shape <= 0')
        # fill value to list
        if len(shape) == 1:
            tmp = [fill_value for i in range(shape[0] - len(l))]
            t = l + tmp
            return t
        # Recursive call _create_list_as_shape
        res = []
        for i in l:
            one = self._create_list_as_shape(i, shape[1:])
            res += [one]
        # add fill list
        if len(l) < shape[0]:
            for i in range(shape[0] - len(l)):
                res += [self._create_list_as_shape([], shape[1:])]
        elif len(l) > shape[0]:
            raise ListShapeErrorException('dim of list is larger than shape. l_len: {}, shape: {}'.format(len(l), shape[0]))
        return res

    def to_list(self):
        res = self._create_list_as_shape(self.l, self.shape, self.fill_value)
        return res

    def __getitem__(self, item):
        ori = item
        if isinstance(item, int):
            if item < 0:
                item += len(self)
            if item < 0 or item > len(self):
                raise IndexError('The index {} is out of range {}'.format(ori, len(self)))
            if len(self.shape) == 1:
                res = self.l[item] if item < len(self.l) else self.fill_value
                return res
            if item >= len(self.l) and item < self.shape[0]:
                return PaddedList([], fill_value=self.fill_value, shape=self.shape[1:])
            elif item >= self.shape[0]:
                raise IndexError('index out of range. list length: {}, i: {}'.format(self.shape[0], item))
            return PaddedList(self.l[item], fill_value=self.fill_value, shape=self.shape[1:])
        elif isinstance(item, slice):
            # len(self.l) == shape[0] should be True. In other word, the first dim should be full.
            tmp_sli = [self.l[ii] for ii in range(*item.indices(len(self)))]
            tmp_shape = [len(tmp_sli)] + self.shape[1:]
            return PaddedList(tmp_sli, fill_value=self.fill_value, shape=tmp_shape)
        else:
            raise TypeError('Invalid argument type. except int or slice but fount {}'.format(type(item)))

    def __len__(self):
        return self.shape[0]

    def __contains__(self, item):
        for i in self:
            if i == item:
                return True
        return False

    def __iter__(self):
        if len(self.shape) == 1:
            for i in range(len(self.l)):
                yield self.l[i]
            for i in range(len(self.l), self.shape[0]):
                yield self.fill_value
        else:
            for i in range(len(self.l)):
                yield PaddedList(self.l[i], fill_value=self.fill_value, shape=self.shape[1:])
            for i in range(len(self.l), self.shape[0]):
                yield PaddedList([], fill_value=self.fill_value, shape=self.shape[1:])

    def __reversed__(self):
        l_len = len(self.l)
        if len(self.shape) == 1:
            for i in range(l_len, self.shape[0]):
                yield self.fill_value
            for i in range(l_len):
                yield self.l[l_len - i - 1]
        else:
            for i in range(l_len, self.shape[0]):
                yield PaddedList([], fill_value=self.fill_value, shape=self.shape[1:])
            for i in range(l_len):
                yield PaddedList(self.l[l_len - i - 1], fill_value=self.fill_value, shape=self.shape[1:])

    def __eq__(self, other):
        if isinstance(other, PaddedList):
            if other.l == self.l and other.shape == self.shape and other.fill_value == self.fill_value:
                return True
        return False

    def __ne__(self, other):
        if isinstance(other, PaddedList):
            if other.l == self.l and other.shape == self.shape and other.fill_value == self.fill_value:
                return False
        return True

    def index(self, x, start: int = ..., end: int = ...):
        for i in range(len(self)):
            if self[i] == x:
                return i
        return -1

    def count(self, x):
        cou = 0
        for i in self:
            if i == x:
                cou += 1
        return cou


class ShapeDifferentException(Exception):
    pass


class ListShapeErrorException(Exception):
    pass


def key_transform(transform, *key, ):
    def transform_fn(sample):
        if len(key) == 1:
            sample[key[0]] = transform(sample[key[0]])
        else:
            in_sample = {k: sample[k] for k in key}
            res = transform(in_sample)
            for k in key:
                del sample[k]
            sample = {**sample, **res}

        # print("sample:{}".format(sample))
        return sample

    return transform_fn


class CopyMap(object):
    def __call__(self, sample):
        return copy.copy(sample)


class IsNone(object):
    def __init__(self, name):
        self._name = name

    def __call__(self, sample):
        if sample is None:
            print("{} is None".format(self._name))
        return sample


class FlatMap(object):
    """
    This map the sample dict to a flat map
    """
    def __call__(self, sample: dict):
        res = {}

        def add_(d: dict):
            for k, v in d.items():
                if not isinstance(v, dict):
                    res[k] = v
                else:
                    add_(v)
        add_(sample)
        return res


def index_select(seq: typing.List, index: typing.List[int]):
    return [seq[k] for k in index]


def filter_token_ids(token_ids, start, end, unk):

    def filter_special_token(token_ids, val):
        return list(filter(lambda x: x != val, token_ids))

    try:
        end_position = token_ids.index(end)
        token_ids = token_ids[:end_position]
    except ValueError as e:
        end_position = None
    token_ids = filter_special_token(token_ids, start)
    token_ids = filter_special_token(token_ids, end)
    token_ids = filter_special_token(token_ids, unk)
    return token_ids, end_position

def convert_one_token_ids_to_code(token_ids, id_to_word_fn, start, end, unk, includes=None):
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    token_ids, _ = filter_token_ids(token_ids, start, end, unk)
    tokens = [id_to_word_fn(tok) for tok in token_ids]
    code = ' '.join(tokens)
    for inc in includes:
        code = (inc + '\n') + code
    return code


def compile_syntax_c_code_by_gcc(code, file_path):
    write_code_to_file(code, file_path)
    # res = os.system('gcc -fsyntax-only -pedantic-errors -std=gnu99 {} >/dev/null 2>/dev/null'.format(file_path))
    res = os.system('gcc -c -pedantic-errors -std=gnu99 {} >/dev/null 2>/dev/null'.format(file_path))
    if res == 0:
        return True
    return False


def compile_c_code_by_gcc(code, file_path):
    write_code_to_file(code, file_path)
    res = os.system('gcc -pedantic-errors -std=gnu99 {} >/dev/null 2>/dev/null'.format(file_path))
    # res = os.system('gcc -pedantic-errors -std=gnu99 {}'.format(file_path))
    if res == 0:
        return True
    return False


def compile_c_code_by_gcc_c89(code, file_path):
    write_code_to_file(code, file_path)
    res = os.system('gcc -pedantic-errors -std=gnu89 {} >/dev/null 2>/dev/null'.format(file_path))
    # res = os.system('gcc -pedantic-errors -std=gnu89 {}'.format(file_path))
    if res == 0:
        return True
    return False


def compile_cpp_code_by_gcc(code, file_path):
    write_code_to_file(code, file_path)
    # res = os.system('g++ -c -pedantic-errors -std=gnu99 {} >/dev/null 2>/dev/null'.format(file_path))
    res = os.system('g++ {} >/dev/null 2>/dev/null'.format(file_path))
    # res = os.system('g++ {}'.format(file_path))
    # print('g++ -I/usr/local/include -std=gnu++98 {}'.format(file_path))
    if res == 0:
        return True
    return False


def write_code_to_file(code, file_path):
    file_path = os.path.abspath(file_path)
    ensure_file_path(file_path)
    f = open(file_path, 'w')
    f.write(code)
    f.flush()
    f.close()
    return file_path


def ensure_file_path(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

class MaskList(collections.Sequence):
    def __init__(self, length, default_value):
        self._length = length
        self._default_value = default_value
        self._mask_segment = []

    def __getitem__(self, i: int):
        for a, b in self._mask_segment:
            if a <= i <= b:
                return 1 - self._default_value
        return self._default_value

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        mask_index = 0
        for i in range(len(self)):
            while True:
                if mask_index >= len(self._mask_segment):
                    yield self._default_value
                    break
                elif i < self._mask_segment[mask_index][0]:
                    yield self._default_value
                    break
                elif self._mask_segment[mask_index][0] <= i <= self._mask_segment[mask_index][1]:
                    yield 1 - self._default_value
                    break
                elif i > self._mask_segment[mask_index][1]:
                    mask_index += 1
                    continue


    def set_mask(self, begin, end=None):
        if end is None:
            end = begin
        self._mask_segment.append((begin, end))

    def flip(self):
        self._default_value = 1 - self._default_value
        return self

    def sort(self):
        self._mask_segment = sorted(self._mask_segment, key=lambda x: x[0])
        return self

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for a, b in zip(self, other):
            if a != b:
                return False
        return True


def transform_id_to_token(one_sequence_ids, id_to_word_fn, offset=0):
    # if not isinstance(one_sequence_ids[0], int):
    #     one_sequence_ids = [i.item() for i in one_sequence_ids]
    # if isinstance(one_sequence_ids, int):
    #     pass
    # else:
    tokens = [id_to_word_fn(i+offset) for i in one_sequence_ids]
    return tokens


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]


def weight_choice(weight):
    """
    :param weight: list对应的权重序列
    :return:选取的值在原列表里的索引
    """
    t = random.uniform(0, sum(weight))
    for i, val in enumerate(weight):
        t -= val
        if t < 0:
            return i


# ----------------- init source code read from codeforce database directly ------------------------------ #

def check_ascii_character(code:str):
    return all(ord(c) < 128 for c in code)


def init_code(code):
    code = code.replace('\ufeff', '').replace('\u3000', ' ')
    code = remove_blank(code)
    code = remove_r_char(code)
    code = remove_comments(code)
    code = remove_blank_line(code)
    return code


def remove_comments(code):
    pattern = r"(\".*?(?<!\\)\"|\'.*?(?<!\\)\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, code)


def remove_blank_line(code):
    code = "\n".join([line for line in code.split('\n') if line.strip() != ''])
    return code


def remove_r_char(code):
    code = code.replace('\r', '')
    return code


def remove_blank(code):
    pattern = re.compile('''('.*?'|".*?"|[^ \t\r\f\v"']+)''')
    mat = re.findall(pattern, code)
    processed_code = ' '.join(mat)
    return processed_code


# ------------------ lex token util method ----------------------- #
def build_code_string_from_lex_tokens(tokens):
    """
    This function build the original code string from the token iterator
    :param tokens: Token iterator
    :return: code string
    """
    lex_tokens = iter(tokens)
    code_re = ""
    lino_pre = 1
    lexpos_pre = 0
    lexpos_temp = 0
    lenth_v = 0
    for token in lex_tokens:
        lino_temp = token.lineno
        if (lino_temp != lino_pre):
            code_re = code_re + "\n"*(lino_temp - lino_pre)
            lenth_v = lino_temp - lino_pre + 1
        else:
            code_re = code_re
        lino_pre = token.lineno
        lexpos_temp = token.lexpos
        code_re = code_re + " " * (lexpos_temp - lexpos_pre - lenth_v)
        code_re = code_re + str(token.value)
        lexpos_pre = lexpos_temp
        lenth_v = len(str(token.value))

    print(code_re)
    return code_re


def iterate_directory(root_path, extensions=None, recursive=False):
    """
    iterate file in directory
    :param root_path:
    :param extensions:
    :param recursive:
    :return:
    """

    if not recursive:
        for file in os.listdir(root_path):
            file_name, extension_name = os.path.splitext(file)
            if extensions is None\
                    or ((isinstance(extensions, list) or isinstance(extensions, tuple)) and extension_name in extensions) \
                    or (isinstance(extensions, str) and extensions == extension_name):
                yield os.path.join(root_path, file), file
    else:
        for dir_path, dir_names, file_names in os.walk(root_path):
            for file in file_names:
                file_name, extension_name = os.path.splitext(file)
                if extensions is None \
                        or ((isinstance(extensions, list) or isinstance(extensions, tuple)) and extension_name in extensions) \
                        or (isinstance(extensions, str) and extensions == extension_name):
                    yield os.path.join(dir_path, file), file


def create_sequence_node_link(begin_idx, length):
    idx = np.arange(begin_idx, begin_idx + length - 1, dtype=np.float)
    idy = idx + 1
    self_id = np.arange(begin_idx, begin_idx + length, dtype=np.float)
    return [idx, idy, self_id], [idy, idx, self_id]


def create_distance_node_matrix(length):
    s = np.arange(length)
    a, b = np.meshgrid(s)
    distance = np.abs(a-b)
    distance[np.diag_indices(distance.shape[0])] = np.eye(distance.shape[0])
    distance -= 1
    return distance


def create_sentence_pair_same_node_matrix(s1, s1_begin, s2, s2_begin):
    idx = []
    idy = []
    data = []
    for i, t1 in enumerate(s1):
        for j, t2 in enumerate(s2):
            if t1 == t2:
                idx.append(i+s1_begin)
                idy.append(j++s2_begin)
                data.append(1)
    return idx+idy, idy+idx, data+data

