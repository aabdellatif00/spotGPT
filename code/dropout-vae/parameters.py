import torch 
import nltk

epochs = 25
batch_size = 128
vocabulary_size = 5160
hidden_layer_size = 256
embedding_size = 300
latent_size = 100
encoder_layers = 3
decoder_layers = 4
rec_coef = 10
kld_coef = 0.001
learning_rate = 0.0001

#Special word tokens
unk_token = "<unk>"
pad_token = "<pad>"
start_token = "<sos>"
end_token = "<eos>"

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def tokenize(text):                   
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokenized_sentence = tokenizer.tokenize(text)
    return tokenized_sentence

