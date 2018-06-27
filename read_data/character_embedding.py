import abc

import config
from common import util
import functools
import more_itertools
import itertools
import tensorflow as tf
import numpy as np


class CharacterEmbedding(object):

    def __init__(self, token_set: set, n_gram=1, ):
        """
        :param token_set: a set of all characters
        """
        self.BEGIN = "<BEGIN>"
        self.END = "<END>"
        self.UNK = "<UNK>"
        self.PAD = "<PAD>"
        self.BEGIN_TOKEN = "<BEGIN_TOKEN>"
        self.END_TOKEN = "<END_TOKEN>"
        self.n_gram = n_gram
        token_set = set(more_itertools.flatten(map(lambda x: list(self.preprocess_token(x)), token_set)))
        token_set = sorted(list(token_set))
        self.id_to_character_dict = dict(list(enumerate(start=0, iterable=token_set)))
        self.id_to_character_dict[len(self.id_to_character_dict)] = self.UNK
        self.id_to_character_dict[len(self.id_to_character_dict)] = self.PAD
        self.id_to_character_dict[len(self.id_to_character_dict)] = self.BEGIN_TOKEN
        self.id_to_character_dict[len(self.id_to_character_dict)] = self.END_TOKEN
        self.character_to_id_dict = util.reverse_dict(self.id_to_character_dict)

    def preprocess_token(self, x):
        return more_itertools.windowed([self.BEGIN] + list(x) + [self.END], self.n_gram)

    def preprocess_token_without_label(self, x):
        return more_itertools.windowed(list(x), self.n_gram)

    def parse_string(self, string_list):
        """
        :param string_list: a list of list of tokens
        :return: a list of list of list of characters of tokens
        """
        max_string_len = max(map(lambda x: len(x)+2, more_itertools.collapse(string_list)))
        max_text_len = max(map(lambda x:len(x), string_list))
        print("max string len:{}".format(max_string_len))
        print("max text len:{}".format(max_text_len))
        def parse_toke(token):
            token = self.preprocess_token(token)
            token = [self.character_to_id_dict[c] for c in token]
            token = token+[0]*(max_string_len-len(token))
            return token

        string_list = [[parse_toke(t) for t in l] for l in string_list]
        empty_token = [0]*max_string_len
        string_list = [l+list(itertools.repeat(empty_token, times=max_text_len-len(l))) for l in string_list]
        return string_list

    def parse_token(self, token, character_position_label=True):
        if character_position_label:
            token = self.preprocess_token(token)
        else:
            token = self.preprocess_token_without_label(token)
        token = [self.character_to_id_dict[c] if c in self.character_to_id_dict else self.character_to_id_dict[self.UNK]
                 for c in token]
        return token

    def parse_string_without_padding(self, string_list, character_position_label=True):
        '''
        parse string list to a char list
        :param string_list: a list of list of tokens
        :return: a list of list of list of characters of tokens
        '''
        string_list = [[self.parse_token(token, character_position_label) for token in l] for l in string_list]
        string_list = [[[self.character_to_id_dict[self.BEGIN_TOKEN]]] + l + [[self.character_to_id_dict[self.END_TOKEN]]]
                       for l in string_list]
        return string_list


def load_character_vocabulary(n_gram, token_list) -> CharacterEmbedding:
    token_set = set(more_itertools.collapse(token_list))
    return CharacterEmbedding(token_set, n_gram)

