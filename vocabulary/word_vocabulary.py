from common import util


class Vocabulary(object):
    def __init__(self,
                 word_set:set,
                 word_to_id_dict: dict,
                 begin_tokens,
                 end_tokens,
                 unk_token,
                 pad_token=None,
                 addition_tokens=None,
                 add_position_to_dict=True):
        self.unk = unk_token
        self.pad = pad_token
        self.begin_tokens = begin_tokens
        self.end_tokens = end_tokens
        self.addition_tokens = addition_tokens if addition_tokens is not None else []
        if add_position_to_dict:
            position_tokens = set(begin_tokens)
            position_tokens |= set(end_tokens)
            if pad_token is not None:
                position_tokens |= {pad_token}
            position_tokens |= {unk_token}
            position_tokens |= set(self.addition_tokens)
            self.word_set = word_set | position_tokens
            for token in sorted(position_tokens):
                word_to_id_dict[token] = len(word_to_id_dict)
        self.word_to_id_dict = word_to_id_dict
        self.id_to_word_dict = util.reverse_dict(self.word_to_id_dict)

    def word_to_id(self, word):
        if word in self.word_to_id_dict.keys():
            return self.word_to_id_dict[word]
        else:
            return self.word_to_id_dict[self.unk]

    def id_to_word(self, i):
        if i not in self.id_to_word_dict.keys():
            print('')
        return self.id_to_word_dict[i]

    def parse_text(self, texts, use_position_label = False):
        """
        :param texts: a list of list of token
        :param use_position_label: whether add begin and end token
        :return:
        """
        if use_position_label:
            texts = [self.begin_tokens + text + self.end_tokens for text in texts]
        max_text = max(map(lambda x:len(x), texts))
        texts = [[self.word_to_id(token) for token in text] for text in texts]
        texts = [text+[0]*(max_text-len(text)) for text in texts]
        return texts

    def parse_text_without_pad(self, texts, use_position_label=False):
        if use_position_label:
            texts = [self.begin_tokens + text + self.end_tokens for text in texts]
        texts = [[self.word_to_id(token) for token in text] for text in texts]
        return texts

    @property
    def vocabulary_size(self):
        return len(self.id_to_word_dict)


def load_vocabulary(load_vocabulary_fn, load_vocabulary_id_dict, begin_tokens ,end_tokens, unk_token, pad_token, addition_tokens) -> Vocabulary:
    return Vocabulary(load_vocabulary_fn(), load_vocabulary_id_dict(), begin_tokens, end_tokens, unk_token, pad_token, addition_tokens)
