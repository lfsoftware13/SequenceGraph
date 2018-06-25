import more_itertools
from nltk.tokenize import StanfordTokenizer
from nltk.tokenize.casual import TweetTokenizer
import typing

from . import util


def stanford_tokenize(texts: typing.List[str]) -> typing.List[typing.List[str]]:
    """
    This function takes string list and then tokenize every str in the list.
    """
    tokenizer = StanfordTokenizer()
    return tokenizer.tokenize_sents(texts)


def tweet_tokenize(texts: typing.List[str]) -> typing.List[typing.List[str]]:
    """
    This function takes string list and then tokenize every str in the list.
    """
    tokenizer = TweetTokenizer()
    def tok(t):
        try:
            r = tokenizer.tokenize(t)
        except Exception as e:
            print("error {} happened at {}".format(e, t))
            raise e
        return r

    return [tok(text) for text in texts]


def parallel_tokenize(core_number, tokenize, texts):
    partition = lambda x: [list(t) for t in more_itertools.divide(core_number, x)]
    r = util.parallel_map(core_number, tokenize, partition(texts))
    return list(more_itertools.flatten(r))