import os
import numpy as np
import openai
import pandas as pd

from dotenv import load_dotenv

# API KEYS
load_dotenv()


def openai_setup():
    openai.organization = "org-DAKfxuGuaoVJNaFffnWZ3Ueu"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    (openai.Model.list())
openai_setup()

def get_log_probability(text):
    kwargs = {"engine": 'text-curie-001',
              "temperature": 0,
              "max_tokens": 0,
              "echo": True,
              "logprobs": 100}
    op = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
    result = op['choices'][0]
    tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

    assert len(tokens) == len(
        logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

    return np.mean(logprobs)

# domain for this function should be [1, inf]
def get_likelihood_ratio(real_lp_score, sampled_lp_score):
    return np.exp(real_lp_score - sampled_lp_score)

def generate_text(text):
    return text[len(text) - 1 : 20] + text[0: 20]

openai_setup()
print(get_log_probability(    "Sexhow railway station was a railway station located in the town of Sexhow, on the Cumbrian Coast Line in North West England. The station was opened by the Lancashire and Yorkshire Railway on 7 October 1870. It was closed to passengers on 5 January 1950, and to goods on 12 May 1965. The station building is now a private residence. There is a small amount of trackage remaining near the building, used currently by a local agricultural business."))
print(get_log_probability("<sos> river station was a railway station located in the town of <unk> on the <unk> <unk> <unk> in <unk> <unk> <unk> <unk> station was opened by the <unk> and <unk> <unk> on 7 <unk> <unk> <unk> was closed to else on 5 <unk> <unk> and to goods on 12 <unk> <unk> <unk> station building is now a private <unk> <unk> is a small amount of <unk> remaining near the <unk> used currently by a local agricultural <unk>"))

# hugging_face_data = pd.read_csv("../../data/GPT-wiki-intro.csv")

# n, d = (hugging_face_data.shape)

# hugging_face_data['split'] = np.random.randn(hugging_face_data.shape[0], 1)

# mask = np.random.rand(len(hugging_face_data)) <= 0.85
# test_data = hugging_face_data[~mask]

# shrunk_test_data = test_data[["wiki_intro", "generated_intro"]].head(n = 500)

# plot_dataframe = pd.DataFrame()

# for idx, row in shrunk_test_data.iterrows():
#     mgt_to_add = {'Type' : 'MGT', 'candidate_text' : row["generated_intro"]}
#     hgt_to_add = {'Type' : 'HGT', 'candidate_text' : row["wiki_intro"]}

#     plot_dataframe = plot_dataframe.append(mgt_to_add, ignore_index = True)
#     plot_dataframe = plot_dataframe.append(hgt_to_add, ignore_index = True)

# probability_ratios = [0] * 1000
# for idx, row in plot_dataframe.head(n = 10).iterrows():
#     text = row['candidate_text']
#     sampled_text = generate_text(text)
#     real_lp = get_log_probability(text)
#     sampled_lp = get_log_probability(sampled_text)
#     print(real_lp, sampled_lp)
#     probability_ratio = get_likelihood_ratio(real_lp, sampled_lp)
#     probability_ratios[idx] = probability_ratio

# plot_dataframe['probability_ratio'] = probability_ratios    
# print(plot_dataframe.head(n = 10))








