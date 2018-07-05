import torch
from torch import nn

from common.torch_util import Attention
from common.word_embedding import WordPairEmbedding, WordEmbedding
from common.highway import Highway


class InputPairEmbedding(nn.Module):
    
    def __init__(self, word_embeddings, word_embed_dim=300,
                highway_n_layers=2, mixed=False):
        
        super(InputPairEmbedding, self).__init__()
        
        self.wordEmbedding = WordPairEmbedding(word_embeddings, mixed=mixed)
        highway_input_size = word_embed_dim * (1 + int(mixed))
        self.highway = Highway(input_size=highway_input_size,
                               n_layers=highway_n_layers)
    
    def forward(self, context_w, question_w):
        context, question = self.wordEmbedding(context_w, question_w)

        context = self.highway(context)
        question = self.highway(question)
        
        return context, question


class InputEmbedding(nn.Module):
    def __init__(self, word_embeddings, word_embed_dim=300,
                highway_n_layers=2, mixed=False, trainable=False):
        super(InputEmbedding, self).__init__()

        self.wordEmbedding = WordEmbedding(word_embeddings, mixed=mixed, trainable=trainable)
        highway_input_size = word_embed_dim * (1 + int(mixed))
        self.highway = Highway(input_size=highway_input_size,
                               n_layers=highway_n_layers)

    def forward(self, text):
        text = self.wordEmbedding(text)
        return self.highway(text)


class RandomInitialInputEmbedding(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, highway_n_layers=2,):
        super().__init__()
        self.wordEmbedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.highway = Highway(input_size=embedding_dim,
                               n_layers=highway_n_layers)

    def forward(self, x):
        o = self.wordEmbedding(x)
        o = self.highway(o)
        return o


class InputEmbeddingWithMatchTag(nn.Module):

    def __init__(self, word_embeddings, word_embed_dim=300,
                 highway_n_layers=2, mixed=False):
        super().__init__()

        self.wordEmbedding = WordPairEmbedding(word_embeddings, mixed=mixed)
        highway_input_size = word_embed_dim * (1 + int(mixed)) + 1
        self.highway = Highway(input_size=highway_input_size,
                               n_layers=highway_n_layers)

    def forward(self, context_w, context_match_tag, question_w, question_match_tag):
        context, question = self.wordEmbedding(context_w, question_w)
        context = torch.car((context, context_match_tag.unsqueeze(-1)), dim=-1)
        question = torch.car((question, question_match_tag.unsqueeze(-1)), dim=-1)

        context = self.highway(context)
        question = self.highway(question)

        return context, question


class WordCharInputEmbedding(nn.Module):
    def __init__(self, word_embeddings, char_embeddings,
                 highway_n_layers=2, mixed=False):
        super().__init__()
        assert word_embeddings.shape[1] == char_embeddings.shape[1]
        self.word_input_embedding = InputPairEmbedding(word_embeddings, word_embeddings.shape[1], highway_n_layers, mixed)
        self.char_input_embedding = InputPairEmbedding(char_embeddings, char_embeddings.shape[1], highway_n_layers, mixed)
        if mixed:
            self.attention = Attention(word_embeddings.shape[1]*2,)
        else:
            self.attention = Attention(word_embeddings.shape[1])

    def forward(self, q1_word, q1_char, q2_word, q2_char):
        q1_word_embedding, q2_word_embedding = self.word_input_embedding(q1_word, q2_word)
        q1_char_embedding, q2_char_embedding = self.word_input_embedding(q1_char, q2_char)
        _, _, q1_char_embedding = self.attention(q1_word_embedding, q1_char_embedding)
        _, _, q2_char_embedding = self.attention(q2_word_embedding, q2_char_embedding)
        return torch.cat((q1_word_embedding, q1_char_embedding), dim=-1), \
               torch.cat((q2_word_embedding, q2_char_embedding), dim=-1)


