import torch
from torch import nn

from qanet.word_embedding import WordEmbedding
from qanet.highway import Highway


class InputEmbedding(nn.Module):
    
    def __init__(self, word_embeddings, word_embed_dim=300,
                highway_n_layers=2):
        
        super(InputEmbedding, self).__init__()
        
        self.wordEmbedding = WordEmbedding(word_embeddings)
        self.highway = Highway(input_size = word_embed_dim,
                               n_layers=highway_n_layers)
    
    def forward(self, context_w, question_w):
        context, question = self.wordEmbedding(context_w, question_w)

        context = self.highway(context)
        question = self.highway(question)
        
        return context, question
