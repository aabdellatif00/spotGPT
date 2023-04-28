import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data
import torchtext
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from collections import Counter
import os
from pathlib import Path

os.chdir(Path(__file__).parent.resolve())


import nltk
import numpy as np
import matplotlib.pyplot as plt
from parameters import learning_rate, tokenize, to_cuda, vocabulary_size, embedding_size, hidden_layer_size, hidden_layers, latent_size, start_token, end_token, batch_size, kld_coef, rec_coef
from .dropout_encoder import Encoder
from .dropout_decoder import Decoder


#Dataset class with function to prepare dataset
class MyDataset(data.Dataset):
    def __init__(self, path, text_field, **kwargs):
        fields = [('text', text_field)]
        examples = []
        with open(path, 'r', encoding='utf-8') as file:
            for text in file:
                examples.append(data.Example.fromlist([text], fields))
        super(MyDataset, self).__init__(examples, fields, **kwargs)
    @classmethod
    def splits(cls, text_field, train='train', **kwargs):
        return super(MyDataset, cls).splits(text_field=text_field, train=train, **kwargs)
                             

#VAE class
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.encoder = Encoder()
        self.find_mean = nn.Linear(2 * hidden_layer_size, latent_size)
        self.find_log_variance = nn.Linear(2 * hidden_layer_size, latent_size)
        self.decoder = Decoder()
        self.latent_size = latent_size

    def forward(self, x, decoder_input, z = None, decoder_hidden = None):
        if z is None:	                                               
            batch_size, sentence_size = x.size()
            x = self.embedding(x)	                                   
            encoder_hidden_output = self.encoder(x)
            
            mean_out = self.find_mean(encoder_hidden_output)	                       
            log_variance = self.find_log_variance(encoder_hidden_output)	               
            z = to_cuda(torch.randn([batch_size, self.latent_size]))	           
            z = mean_out + z*torch.exp(0.5 * log_variance)	                           
            kld = -0.5*torch.sum(log_variance-mean_out.pow(2)-log_variance.exp()+1, 1).mean()
        else:
            kld = None                                                 

        decoder_input = self.embedding(decoder_input)	                                

        output, decoder_hidden = self.decoder(decoder_input, z, decoder_hidden)
        return output, decoder_hidden, kld