import requests
import pandas as pd
import numpy as np
from ..code.scoring.scoring import get_log_probability

hugging_face_data = pd.read_csv("./GPT-wiki-intro.csv")

train_name = './huggingface-train.txt'
test_name = './huggingface-test.txt'
train = open(train_name, "w")
test = open(test_name, "w")

hugging_face_data['split'] = np.random.randn(hugging_face_data.shape[0], 1)

mask = np.random.rand(len(hugging_face_data)) <= 0.85
train_data = hugging_face_data[mask]
test_data = hugging_face_data[~mask]


for mgt_sentence in train_data['generated_intro']:
    train.write(mgt_sentence + '\n')

for mgt_sentence in test_data['generated_intro']:
    test.write(mgt_sentence + '\n')


train.close()
test.close()
