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
from utils.dropout_vae.parameters import learning_rate, tokenize, to_cuda, vocabulary_size, embedding_size, hidden_layer_size, hidden_layers, latent_size, start_token, end_token, batch_size, kld_coef, rec_coef
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

# train_loss_list = []
# val_loss_list = []
# train_KL_list = []
# val_KL_list = []

# save_path = "saved_models/dropout_vae_model.tar"
# if not os.path.exists("saved_models"):
#     os.makedirs("saved_models")

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)    

# text_field = data.Field(init_token=start_token, eos_token=end_token, lower=True, tokenize=tokenize, batch_first=True)
# train_data, val_data = MyDataset.splits(path="../../../data", train = "huggingface-train.txt", test="huggingface-test.txt", text_field=text_field)
# text_field.build_vocab(train_data, val_data, max_size=vocabulary_size-4, vectors = 'glove.6B.300d')
# vocab = text_field.vocab
# train_iter, val_iter = data.BucketIterator.splits((train_data, val_data), batch_size=batch_size, sort_key = lambda x: len(x.text), repeat = False, device = torch.device('cpu'))
    

# vae = VAE()
# weight_matrix = vocab.vectors
# vae.embedding.weight.data.copy_(weight_matrix)            
# vae = to_cuda(vae)

# optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# def train_batch(x, decoder_input, step, train = True):
#     output, _, kld = vae(x, decoder_input, None, None)
#     output = output.view(-1, vocabulary_size)	               
#     x = x.contiguous().view(-1)	                           
#     rec_loss = F.cross_entropy(output, x)
#     loss = rec_coef*rec_loss + kld_coef*kld
#     if train == True:	                                   
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     return rec_loss.item(), kld.item()

# def training():
#     step = 0
#     for epoch in range(1):
#         print(epoch)
#         vae.train()
#         train_rec_loss = []
#         train_kl_loss = []
#         for batch in train_iter:
#             x = batch.text 	                             
#             decoder_input = x
#             rec_loss, kl_loss = train_batch(x, decoder_input, step, train=True)
#             train_rec_loss.append(rec_loss)
#             train_kl_loss.append(kl_loss)
#             step += 1

#         vae.eval()
#         valid_rec_loss = []
#         valid_kl_loss = []
#         for batch in val_iter:
#             x = batch.text
#             decoder_input = x
#             with torch.autograd.no_grad():
#                 rec_loss, kl_loss = train_batch(x, decoder_input, step, train=False)
#             valid_rec_loss.append(rec_loss)
#             valid_kl_loss.append(kl_loss)

#         train_rec_loss = np.mean(train_rec_loss)
#         train_kl_loss = np.mean(train_kl_loss)
#         valid_rec_loss = np.mean(valid_rec_loss)
#         valid_kl_loss = np.mean(valid_kl_loss)

#         print("Epoch -> ", epoch)
#         print("Train data -> Reconstruction loss = ", train_rec_loss,", KL divergence = ", train_kl_loss)
#         print("Validation data -> Reconstruction loss = ", valid_rec_loss,", KL divergence = ", valid_kl_loss)
#         train_loss_list.append(train_rec_loss)
#         train_KL_list.append(train_kl_loss)
#         val_loss_list.append(valid_rec_loss)
#         val_KL_list.append(valid_kl_loss)
#         if epoch%5==0:
#             torch.save({
#                 'epoch': epoch + 1,
#                 'vae_dict': vae.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'step': step
#             }, save_path)

# def generate_sentence(input):
#     checkpoint = torch.load(save_path)
#     vae.load_state_dict(checkpoint['vae_dict'])
#     vae.eval()
#     del checkpoint
#     inp = torch.tensor([[vocab.stoi[i] for i in input.split()]])
#     inp = to_cuda(inp)
#     output, _, kld = vae(inp, inp, None, None)
#     probs = F.softmax(output[0], dim=1)
#     final_out = torch.multinomial(probs,1)
#     str = ""
#     for i in final_out:
#         next_word = vocab.itos[i.item()] 
#         str += next_word + " "
#     print(str)

# if __name__ == '__main__':
#     training()
#     print("Input sentence:")
#     print("not all movies are wonderful")
#     print("Output sentence:")
#     generate_sentence("not all movies are wonderful")
#     print("Input sentence:")
#     print("spending some time at seashore can relax us by a great extend")
#     print("Output sentence:")
#     generate_sentence("spending some time at seashore can relax us by a great extend")
    
#     plt.plot(train_loss_list, '-bx')
#     plt.plot(val_loss_list, '-rx')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend(['Train', 'Validation'])
#     plt.title('Loss v/s Epochs')
#     plt.savefig('Plot_Loss')
#     plt.clf()

#     plt.plot(train_KL_list, '-bx')
#     plt.plot(val_KL_list, '-rx')
#     plt.xlabel('Epoch')
#     plt.ylabel('KL divergence')
#     plt.legend(['Train', 'Validation'])
#     plt.title('KL divergence v/s Epochs')
#     plt.savefig('Plot_KL')
#     plt.clf()

#     plt.plot(train_KL_list, '-bx')
#     plt.xlabel('Epoch')
#     plt.ylabel('KL divergence')
#     plt.title('KL divergence v/s Epochs (On train data)')
#     plt.savefig('Plot_KL_train')
   
    