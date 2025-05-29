import pandas as pd
import dgl
import json
import numpy as np
import torch
from tqdm import tqdm
from torch import Tensor
from transformers import AutoTokenizer, BertModel, AutoModel
import os
os.environ['DGL_GRAPHBOLT_USE'] = '0'


df = pd.read_csv("dataset/citation-cooked.csv")

sorted_df = df.sort_values(by=['Year'], ascending=False).reset_index(drop=True)
# Select a subset of the data
n_rows = 10
from_ids, to_ids = [], []
cnt = 0
for i, row in sorted_df.iterrows():
    refs = row["References"]
    if pd.isna(refs):
        continue
    refs = refs.split(", ")
    for ref in refs:
        from_ids.append(row["Index"])
        to_ids.append(int(ref))
    cnt += 1
    if cnt > n_rows:
        break

all_ids = list(set(from_ids + to_ids))
all_ids = sorted(all_ids)

new_id = {old: new for new, old in enumerate(all_ids)}
new_df = df[df["Index"].isin(all_ids)]
new_df["Index"] = new_df["Index"].map(new_id)
new_from_ids = [new_id[x] for x in from_ids]
new_to_ids = [new_id[x] for x in to_ids]

g = dgl.graph((torch.tensor(new_from_ids), torch.tensor(new_to_ids)))
g = dgl.to_bidirected(g)

bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")


def get_embedding_bert(texts, model, tokenizer, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        inputs = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        embeddings.append(outputs.pooler_output)
    return torch.cat(embeddings)


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
model = AutoModel.from_pretrained("thenlper/gte-small")

def get_embedding(texts, model, tokenizer, batch_size=64):
    outputs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_inputs = tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        batch_outputs = model(**batch_inputs)
        batch_embeddings = average_pool(batch_outputs.last_hidden_state, batch_inputs['attention_mask'])
        outputs.append(batch_embeddings)
    return torch.cat(outputs)

corpus = new_df["Title"].fillna("") + "\n" + new_df["Abstract"].fillna("")
corpus_ids = new_df["Index"].tolist()
corpus = corpus.tolist()
prompted_corpus = []
for sen in corpus:
    title, abstract = sen.split("\n")
    title = title.strip()
    abstract = abstract.strip()
    if len(title) > 0 and len(abstract) > 0:
        prompt = f"Title: {title}\nAbstract: {abstract}\n"
    elif len(title) > 0:
        prompt = f"Title: {title}\n"
    elif len(abstract) > 0:
        prompt = f"Abstract: {abstract}\n"
    else:
        raise ValueError("Both title and abstract are empty")
    prompted_corpus.append(prompt)

prompted_corpus_dict = {
    i: prompted_corpus[i] for i in range(len(prompted_corpus))
}
with open("dataset/prompted_corpus.json", "w") as f:
    json.dump(prompted_corpus_dict, f, indent=2)


prompts = json.load(open("dataset/prompted_corpus.json"))
all_extracted = []
texts = []
for i in range(len(all_extracted), min(len(all_extracted) + 640, len(prompts))):
    texts.append(prompts[str(i)])
temp_extracted = get_embedding(texts, model, tokenizer)
temp_extracted = temp_extracted.cpu().detach().numpy()
all_extracted = temp_extracted

with open('dataset/extracted_prompts.npy', 'wb') as f:
    np.save(f, all_extracted)

node_features = get_embedding_bert(prompted_corpus, bert_model, bert_tokenizer)
node_features = torch.tensor(np.load("dataset/extracted_prompts.npy"))
g.ndata['feat'] = node_features
from dgl.data.utils import save_graphs
name =  "dataset/citation.dgl"
save_graphs(name, g)
