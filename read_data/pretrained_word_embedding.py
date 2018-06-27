import abc

import more_itertools
import numpy
import numpy as np
import tensorflow as tf
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec

import config
from common import util


class WordEmbedding(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model):
        self.model = model

    def __getitem__(self, item):
        # if item in self.model:
        #     return self.model[item]
        # else:
        #     return numpy.random.randn(*self.model['office'].shape)
        try:
            r = self.model[item]
        except Exception:
            r = numpy.random.randn(*self.model['office'].shape)
        return r


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r', encoding='utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


class GloveWordEmbedding(WordEmbedding):
    def __init__(self):
        super().__init__(loadGloveModel(config.pretrained_glove_path))


class FastTextWordEmbedding(WordEmbedding):
    def __init__(self):
        super().__init__(FastText.load_fasttext_format(config.pretrained_fasttext_path))
        print("the fasttext model loaded")


class Vocabulary(object):
    def __init__(self, embedding: WordEmbedding, word_set: set, use_position_label: bool, begin_tokens=None, end_tokens=None):
        self.unk = '<unk>'
        self.begin = ['<BEGIN>']
        self.end = ['<END>']
        if begin_tokens is not None:
            self.begin = begin_tokens
        if end_tokens is not None:
            self.end = end_tokens
        self.pad = '<PAD>'
        self.use_position_label = use_position_label
        word_set = sorted(set(word_set))
        self.id_to_word_dict = dict(list(enumerate(word_set, start=2)))
        self.id_to_word_dict[0] = self.unk
        self.id_to_word_dict[1] = self.pad
        if use_position_label:
            for tok in self.begin:
                self.id_to_word_dict[len(self.id_to_word_dict)] = tok
            for tok in self.end:
                self.id_to_word_dict[len(self.id_to_word_dict)] = tok
        self.word_to_id_dict = util.reverse_dict(self.id_to_word_dict)
        print("The word vocabulary has {} words".format(len(self.word_to_id_dict)))
        self._embedding_matrix = np.array([embedding[b] for a, b in sorted(self.id_to_word_dict.items(), key=lambda x:x[0])])

    def word_to_id(self, word):
        if word in self.word_to_id_dict.keys():
            return self.word_to_id_dict[word]
        else:
            return 0

    def id_to_word(self, i):
        if i:
            return self.id_to_word_dict[i]
        else:
            return self.unk

    @property
    def embedding_matrix(self):
        return self._embedding_matrix

    def parse_text(self, texts, position_label_index=0):
        """
        :param texts: a list of list of token
        :return:
        """
        if self.use_position_label:
            texts = [[self.begin[position_label_index]] + text + [self.end[position_label_index]] for text in texts]
        max_text = max(map(lambda x:len(x), texts))
        texts = [[self.word_to_id(token) for token in text] for text in texts]
        texts = [text+[0]*(max_text-len(text)) for text in texts]
        return texts

    def parse_text_without_pad(self, texts, position_label_index=0):
        if self.use_position_label:
            texts = [[self.begin[position_label_index]] + text + [self.end[position_label_index]] for text in texts]
        texts = [[self.word_to_id(token) for token in text] for text in texts]
        return texts

    @property
    def vocabulary_size(self):
        return len(self.id_to_word_dict)


def load_vocabulary(word_vector_name, text_list, use_position_label=False, begin_tokens=None, end_tokens=None) -> Vocabulary:
    namd_embedding_dict = {"glove": GloveWordEmbedding, "fasttext": FastTextWordEmbedding,}
    word_set = more_itertools.collapse(text_list)
    return Vocabulary(namd_embedding_dict[word_vector_name](), word_set, use_position_label=use_position_label, begin_tokens=begin_tokens, end_tokens=end_tokens)