import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data

from utils.dropout_vae.parameters import hidden_layer_size, encoder_layers, embedding_size, to_cuda

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_hidden_size = hidden_layer_size
        self.encoder_layers = encoder_layers
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_layer_size, num_layers=encoder_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)

    def init_hidden(self, batch_size):
        h_init = torch.zeros(2*self.encoder_layers, batch_size, self.encoder_hidden_size)
        c_init = torch.zeros(2*self.encoder_layers, batch_size, self.encoder_hidden_size)
        self.hidden = (to_cuda(h_init), to_cuda(c_init))

    def forward(self, x):
        batch_size, sentence_size, embedded_size = x.size()
        self.init_hidden(batch_size)
        _, (self.hidden, _) = self.lstm(x, self.hidden)	            
        self.hidden = self.dropout(self.hidden)
        self.hidden = self.hidden.view(self.encoder_layers, 2, batch_size, self.encoder_hidden_size)
        self.hidden = self.hidden[-1]	                            
        hidden_output = torch.cat(list(self.hidden), dim=1)	               
        return hidden_output