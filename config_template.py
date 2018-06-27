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

fasttext_dir = 'fasttext'
pretrained_fasttext_path = os.path.join(embedding_path, fasttext_dir, 'wiki.en.bin')

multinli_data_path = r'/home/lf/Project/multinli_1.0/multinli_1.0'
