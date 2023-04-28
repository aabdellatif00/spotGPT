import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data

from utils.dropout_vae.parameters import hidden_layer_size, decoder_layers, embedding_size, to_cuda, latent_size, vocabulary_size

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_hidden_size = hidden_layer_size
        self.decoder_layers = decoder_layers
        self.latent_size = latent_size
        self.lstm = nn.LSTM(input_size=embedding_size+latent_size, hidden_size=hidden_layer_size, num_layers=decoder_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, vocabulary_size)
        self.dropout = nn.Dropout(p=0.3)

    def init_hidden(self, batch_size):
        h_init = torch.zeros(self.decoder_layers, batch_size, self.decoder_hidden_size)
        c_init = torch.zeros(self.decoder_layers, batch_size, self.decoder_hidden_size)
        self.hidden = (to_cuda(h_init), to_cuda(c_init))

    def forward(self, x, z, decoder_hidden = None):
        batch_size, sentence_size, embedded_size = x.size()
        z = torch.cat([z]*sentence_size, 1).view(batch_size, sentence_size, self.latent_size)	
        x = torch.cat([x,z], dim=2)	                                    

        if decoder_hidden is None:	                                    
            self.init_hidden(batch_size)
        else:					                                    
            self.hidden = decoder_hidden

        output, self.hidden = self.lstm(x, self.hidden)
        output = self.dropout(output)
        output = self.fc(output)

        return output, self.hidden