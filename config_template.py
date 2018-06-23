# root path of the project
import os

from common import util

root = r'/home/lf/Project/GrammaLanguageModel'
# tmp path
temp_code_write_path = r'tmp'
# scrapyOJ db path
scrapyOJ_path = r'/home/lf/new_disk/data_store/codeforces/scrapyOJ.db'
# cache path
cache_path = r'/home/lf/Project/GrammaLanguageModel/data/cache_data'
save_model_root = os.path.join(root, 'trained_model')
util.make_dir(save_model_root)
summarization_source_code_to_method_name_path = r'/home/lf/Project/GrammaLanguageModel/data/summarization_method_name/json'