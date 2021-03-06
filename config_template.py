# root path of the project
import os

from common import util

root = r'/home/lf/Project/SequenceGraph'
DATA_PATH = os.path.join(root, 'data')
# tmp path
temp_code_write_path = r'tmp'
# scrapyOJ db path
scrapyOJ_path = r'/home/lf/new_disk/data_store/codeforces/scrapyOJ.db'
# cache path
cache_path = r'/home/lf/Project/SequenceGraph/data/cache_data'
save_model_root = os.path.join(root, 'trained_model')
util.make_dir(save_model_root)
summarization_source_code_to_method_name_path = r'/home/lf/Project/SequenceGraph/data/dataset/json'

embedding_path = os.path.join('/', 'home', 'wlw', 'dataset', 'embedding')
glove_dir = 'glove'
pretrained_glove_path = os.path.join(embedding_path, glove_dir, 'glove.6B', "glove.6B.100d.txt")
pretrained_glove_300d_path = os.path.join(embedding_path, glove_dir, 'glove.840B.300d', 'glove.840B.300d.txt')

fasttext_dir = 'fasttext'
pretrained_fasttext_path = os.path.join(embedding_path, fasttext_dir, 'wiki.en.bin')
pretrained_fasttext_de_path = r'/home/lf/Project/wiki.de/wiki.de.bin'

multinli_data_path = r'/home/lf/Project/multinli_1.0/multinli_1.0'
snli_data_path = r'/home/lf/Project/snli_1.0/snli_1.0'
wmt2014_en_de_path = r'/home/lf/Project/WMT2014_en-de'

sequence_data_path = r'/home/lf/new_disk/sequence_data'
