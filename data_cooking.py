import os
import json
import torch
import dgl
import pandas as pd
from typing import List
from tqdm import tqdm
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BertModel
from dgl.data.utils import save_graphs

# Check and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def build_citation_edges(df: pd.DataFrame):
    """
    Build citation edges (from -> to) from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Index' and 'References' columns.

    Returns:
        Tuple[List[int], List[int], pd.DataFrame, Dict[int, int]]:
            - from_ids: list of source paper indices
            - to_ids: list of target (cited) paper indices
            - new_df: filtered DataFrame containing only papers in the citation graph
            - new_id: mapping from original paper index to new consecutive IDs
    """
    from_ids, to_ids = [], []
    n_rows = 1
    cnt = 0
    for _, row in df.iterrows():
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

    # Create consistent node ID mapping
    all_ids = sorted(set(from_ids + to_ids))
    new_id = {old: new for new, old in enumerate(all_ids)}

    # Filter and remap DataFrame
    new_df = df[df["Index"].isin(all_ids)].copy()
    new_df["Index"] = new_df["Index"].map(new_id)

    new_from_ids = [new_id[x] for x in from_ids]
    new_to_ids = [new_id[x] for x in to_ids]

    return new_from_ids, new_to_ids, new_df, new_id


def create_graph(from_ids: List[int], to_ids: List[int]):
    """
    Create a bidirected DGL graph from citation edges.

    Args:
        from_ids (List[int]): Source node IDs.
        to_ids (List[int]): Destination node IDs.

    Returns:
        dgl.DGLGraph: The constructed graph.
    """
    g = dgl.graph((torch.tensor(from_ids), torch.tensor(to_ids)))
    g = dgl.to_bidirected(g)
    return g


def prepare_prompted_texts(df: pd.DataFrame) -> list:
    """
    Create prompted input text for each paper from Title and Abstract.

    Args:
        df (pd.DataFrame): DataFrame containing 'Title' and 'Abstract' columns.

    Returns:
        List[str]: List of prompted input texts.
    """
    corpus = df["Title"].fillna("") + "\n" + df["Abstract"].fillna("")
    corpus = corpus.tolist()

    prompted = []
    for entry in corpus:
        title, abstract = entry.split("\n")
        title, abstract = title.strip(), abstract.strip()

        if title and abstract:
            prompt = f"Title: {title}\nAbstract: {abstract}"
        elif title:
            prompt = f"Title: {title}"
        elif abstract:
            prompt = f"Abstract: {abstract}"
        else:
            raise ValueError("Both title and abstract are empty.")

        prompted.append(prompt)

    return prompted


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Perform average pooling over non-masked tokens.

    Args:
        last_hidden_states (Tensor): Hidden states from the model.
        attention_mask (Tensor): Attention mask.

    Returns:
        Tensor: Pooled embeddings.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def get_embedding(texts, model, tokenizer, batch_size=16):
    """
    Compute sentence embeddings using transformer encoder.

    Args:
        texts (List[str]): Input text list.
        model: Pretrained transformer model.
        tokenizer: Corresponding tokenizer.
        batch_size (int): Batch size for embedding.

    Returns:
        Tensor: Stacked embeddings for all inputs.
    """
    outputs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="GTE Embedding"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        batch_outputs = model(**inputs)
        embeddings = average_pool(batch_outputs.last_hidden_state, inputs['attention_mask'])
        outputs.append(embeddings.cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def get_embedding_bert(texts, model, tokenizer, batch_size=16):
    """
    Compute sentence embeddings using transformer encoder.

    Args:
        texts (List[str]): Input text list.
        model: Pretrained transformer model.
        tokenizer: Corresponding tokenizer.
        batch_size (int): Batch size for embedding.

    Returns:
        Tensor: Stacked embeddings for all inputs.
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT Embedding"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        embeddings.append(outputs.pooler_output.cpu())
    return torch.cat(embeddings, dim=0)


def main():
    # === Load data ===
    df = pd.read_csv("dataset/citation-cooked.csv")
    sorted_df = df.sort_values(by=['Year'], ascending=False).reset_index(drop=True)

    # === Build graph edges ===
    from_ids, to_ids, new_df, new_id = build_citation_edges(sorted_df)

    # === Create graph ===
    g = create_graph(from_ids, to_ids)

    # === Prepare prompted text ===
    prompted_corpus = prepare_prompted_texts(new_df)
    prompted_corpus_dict = {i: text for i, text in enumerate(prompted_corpus)}
    with open("dataset/prompted_corpus.json", "w") as f:
        json.dump(prompted_corpus_dict, f, indent=2)

    # === Load sentence embedding model ===
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    model = AutoModel.from_pretrained("thenlper/gte-small").to(device)
    model.eval()

    bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device)
    bert_model.eval()

    # === Generate or load embeddings ===
    embedding_path = "dataset/extracted_prompts.pt"
    if os.path.exists(embedding_path):
        print("Loading existing embeddings...")
        all_embeddings = torch.load(embedding_path)
    else:
        # all_embeddings = get_embedding(prompted_corpus, model, tokenizer)
        all_embeddings = get_embedding_bert(prompted_corpus, bert_model, bert_tokenizer)
        torch.save(all_embeddings, embedding_path)

    # === Attach node features and save graph ===
    g.ndata['feat'] = all_embeddings
    save_graphs("dataset/citation_all.dgl", g)
    print("Graph saved at dataset/citation_all.dgl")


if __name__ == "__main__":
    main()
