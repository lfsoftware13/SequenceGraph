import os

import more_itertools

from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from config import wmt2014_en_de_path
from read_data import character_embedding
from read_data.pretrained_word_embedding import load_vocabulary, FastTextWordEmbedding, Vocabulary


@disk_cache(basename='load_en_word_set', directory=CACHE_DATA_PATH)
def load_en_word_set():
    word_path = os.path.join(wmt2014_en_de_path, 'vocab.50K.en')
    with open(word_path, encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip().lower() for line in lines]
    return set(lines)


@disk_cache(basename='load_de_word_set', directory=CACHE_DATA_PATH)
def load_de_word_set():
    word_path = os.path.join(wmt2014_en_de_path, 'vocab.50K.de')
    with open(word_path, encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip().lower() for line in lines]
    return set(lines)


@disk_cache(basename='load_en_vocabulary', directory=CACHE_DATA_PATH)
def load_en_vocabulary(word_vector_name='fasttext'):
    vocab = load_vocabulary(word_vector_name, load_en_word_set(), use_position_label=True)
    return vocab


@disk_cache(basename='load_de_vocabulary', directory=CACHE_DATA_PATH)
def load_de_vocabulary(word_vector_name='fasttext') -> Vocabulary:
    namd_embedding_dict = {"fasttext": FastTextWordEmbedding, }
    word_set = more_itertools.collapse(load_de_word_set())
    return Vocabulary(namd_embedding_dict[word_vector_name](language='de'), word_set, use_position_label=True)


@disk_cache(basename='load_en_character_vocabulary', directory=CACHE_DATA_PATH)
def load_en_character_vocabulary(n_gram):
    word_set = load_en_word_set()
    character_vocabulary = character_embedding.load_character_vocabulary(n_gram=n_gram, token_list=word_set)
    return character_vocabulary


@disk_cache(basename='load_de_character_vocabulary', directory=CACHE_DATA_PATH)
def load_de_character_vocabulary(n_gram):
    word_set = load_de_word_set()
    character_vocabulary = character_embedding.load_character_vocabulary(n_gram=n_gram, token_list=word_set)
    return character_vocabulary


if __name__ == '__main__':
    en_vocab = load_en_vocabulary()
    de_vocab = load_de_vocabulary()
    word_set = load_en_word_set()
    print(word_set)
    # de_set = load_de_vocabulary()
    word_id = en_vocab.word_to_id('ingot')
    count = 0
    for k in en_vocab.word_to_id_dict.keys():
        if k[:5] == 'ingot':
            print(k)
        #     count += 1
            # if count < 100:
            #     print(k)
    print(word_id)
