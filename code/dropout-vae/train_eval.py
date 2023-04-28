import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from collections import Counter
import seaborn as sns
import re
import io
import datasets
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from parameters import vocabulary_size, start_token, end_token, kld_coef, rec_coef, to_cuda, make_tokens, epochs, batch_size
from model import VAE, VAEDataset
from scoring import get_likelihood_ratio, get_log_probability


print(torch.cuda.is_available())
print(torch.cuda.device_count())

loss_data = np.ndarray((epochs, 4))

save_path = "saved_models/dropout_vae_model.tar"
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)    

text_field = data.Field(init_token=start_token, eos_token=end_token, lower=True, tokenize=make_tokens, batch_first=True)
train_data, val_data = VAEDataset.splits(path="data", train = "huggingface-train.txt", test="huggingface-test.txt", text_field=text_field)
text_field.build_vocab(train_data, val_data, max_size=vocabulary_size-4, vectors = 'glove.6B.300d')
vocab = text_field.vocab
train_iter, val_iter = data.BucketIterator.splits((train_data, val_data), batch_size=batch_size, sort_key = lambda x: len(x.text), repeat = False, device = torch.device('cuda'))
    

vae = VAE()
weight_matrix = vocab.vectors
vae.embedding.weight.data.copy_(weight_matrix)            
vae = to_cuda(vae)

optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

def train_batch(x, decoder_input, step, train = True):
    output, _, kld = vae(x, decoder_input, None, None)
    output = output.view(-1, vocabulary_size)	               
    x = x.contiguous().view(-1)	                           
    rec_loss = F.cross_entropy(output, x)
    loss = rec_coef*rec_loss + kld_coef*kld
    if train == True:	                                   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return rec_loss.item(), kld.item()

def train_VAE():
    step = 0
    for epoch in range(epochs):
        vae.train()
        train_rec_loss = []
        train_kl_loss = []
        
        for batch in train_iter:
            x = batch.text 	                             
            decoder_input = x
            rec_loss, kl_loss = train_batch(x, decoder_input, step, train=True)
            train_rec_loss.append(rec_loss)
            train_kl_loss.append(kl_loss)
            step += 1
        vae.eval()
        valid_rec_loss = []
        valid_kl_loss = []
        for batch in val_iter:
            x = batch.text
            decoder_input = x
            with torch.autograd.no_grad():
                rec_loss, kl_loss = train_batch(x, decoder_input, step, train=False)
            valid_rec_loss.append(rec_loss)
            valid_kl_loss.append(kl_loss)

        train_rec_loss = np.mean(train_rec_loss)
        train_kl_loss = np.mean(train_kl_loss)
        valid_rec_loss = np.mean(valid_rec_loss)
        valid_kl_loss = np.mean(valid_kl_loss)
        train_loss = train_rec_loss + train_kl_loss
        valid_loss = valid_rec_loss + valid_kl_loss

        print("Epoch -> ", epoch)
        print("Train data -> Objective loss = ", train_loss,", KL divergence = ", train_kl_loss)
        print("Validation data -> Objective loss = ", valid_loss,", KL divergence = ", valid_kl_loss)
        loss_data[epoch] = [train_loss, train_kl_loss, valid_loss, valid_kl_loss]
        
        if epoch%5==0:
            torch.save({
                'epoch': epoch + 1,
                'vae_dict': vae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, save_path)

def generate_reconstruction(input):
    checkpoint = torch.load("saved_models/dropout_vae_model_3_4.tar")
    vae.load_state_dict(checkpoint['vae_dict'])
    vae.eval()
    del checkpoint
    inp = torch.tensor([[vocab.stoi[i] for i in input.split()]])
    inp = to_cuda(inp)
    output, _, kld = vae(inp, inp, None, None)
    probs = F.softmax(output[0], dim=1)
    final_out = torch.multinomial(probs,1)
    str = ""
    for i in final_out:
        next_word = vocab.itos[i.item()] 
        str += next_word + " "
    return str

train_VAE()
plt.plot(loss_data[:, 0], '-bx')
plt.plot(loss_data[:, 1], '-rx')
plt.xlabel('Epoch')
plt.ylabel('Total VAE Objective')
plt.legend(['train', 'val'])
plt.title('Objective Function v/s Epochs')
plt.savefig('Plot_Loss_3_4')
plt.clf()

plt.plot(loss_data[:, 2], '-bx')
plt.plot(loss_data[:, 3], '-rx')
plt.xlabel('Epoch')
plt.ylabel('KL loss')
plt.legend(['train', 'val'])
plt.title('Kullback-Leibler Divergence loss v/s Epochs')
plt.savefig('Plot_KL_3_4')
plt.clf()

    
hugging_face_data = datasets.load_dataset("aadityaubhat/GPT-wiki-intro", split = "train")
hugging_face_data = pd.DataFrame(hugging_face_data)
n, d = (hugging_face_data.shape)

mask = np.random.rand(len(hugging_face_data)) <= 0.85
test_data = hugging_face_data[~mask]

shrunk_test_data = test_data[['wiki_intro', 'generated_intro']].head(n = 500)
shrunk_test_data.head()

plot_dataframe = pd.DataFrame()

for idx, row in shrunk_test_data.iterrows():
    mgt_to_add = {'Type' : 'MGT', 'candidate_text' : row["generated_intro"]}
    hgt_to_add = {'Type' : 'HGT', 'candidate_text' : row["wiki_intro"]}

    plot_dataframe = plot_dataframe.append(mgt_to_add, ignore_index = True)
    plot_dataframe = plot_dataframe.append(hgt_to_add, ignore_index = True)

probability_ratios = [0] * 1000
for idx, row in plot_dataframe.head.iterrows():
    text = row['candidate_text']
    sampled_text = generate_reconstruction(text)
    real_lp = get_log_probability(text)
    sampled_lp = get_log_probability(sampled_text)
    probability_ratio = get_likelihood_ratio(real_lp, sampled_lp)
    probability_ratios[idx] = probability_ratio
plot_dataframe['probability_ratio'] = probability_ratios



fig, ax = plt.subplots()

sns.kdeplot(data = plot_dataframe, 
                x = "probability_ratio", 
                hue = "Type",
                fill=True, 
                common_norm=False, 
                palette="flare",
                alpha=.5, 
                linewidth=0,
                ax = ax
).set(title='Density of Probability Ratio Scores (VAE under curie)', 
      xlabel='Probability Ratio Score', 
      ylabel='Density')
ax.set_xlim(0, 5)
plt.legend(title='Text Type', loc='upper right', labels=['Machine Generated', 'Human Generated'])
plt.savefig("density_estimate_curie.png")