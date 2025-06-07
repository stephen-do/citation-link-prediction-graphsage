import pandas as pd
import networkx as nx
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1. Read and preprocess data ==========
df = pd.read_csv("dataset/citation-cooked.csv")
df = df.dropna(subset=["Index"])
df["Index"] = df["Index"].astype(int)
df = df.head(1000)  # Take only first 1000 rows for faster processing

# Parse References column into list of integers
def parse_references(ref_str):
    if pd.isna(ref_str) or ref_str == "None":
        return []
    return [int(r.strip()) for r in str(ref_str).split(',') if r.strip().isdigit()]

df["References"] = df["References"].apply(parse_references)

# ========== 2. Generate all possible paper pairs (excluding self-pairs) ==========
paper_ids = df["Index"].tolist()
pairs = [(i, j) for i, j in product(paper_ids, repeat=2) if i != j]
df_pairs = pd.DataFrame(pairs, columns=["refer_paper_id", "unique_paper_id"])

# Create dictionaries for fast lookup of References, Authors, Year, and Title
ref_dict = df.set_index("Index")["References"].to_dict()
author_dict = df.set_index("Index")["Authors"].fillna("").apply(
    lambda x: set(a.strip() for a in x.split(",") if a.strip())
).to_dict()
year_dict = df.set_index("Index")["Year"].to_dict()
title_dict = df.set_index("Index")["Title"].fillna("").to_dict()

# ========== 3. Label forward citation: 1 if unique_paper cites refer_paper, else 0 ==========
def has_reference(row):
    refs = ref_dict.get(row["unique_paper_id"], [])
    return int(row["refer_paper_id"] in refs)

df_pairs["forward_refer"] = df_pairs.apply(has_reference, axis=1)

# ========== 4. Calculate TF-IDF similarity between paper titles ==========
titles = [title_dict[i] for i in paper_ids]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(titles)

# Map paper IDs to indices in the TF-IDF matrix
id_to_pos = {pid: idx for idx, pid in enumerate(paper_ids)}

def get_tfidf_sim(row):
    pos1, pos2 = id_to_pos[row["refer_paper_id"]], id_to_pos[row["unique_paper_id"]]
    return cosine_similarity(tfidf_matrix[pos1], tfidf_matrix[pos2])[0][0]

df_pairs["tf_idf_similarity"] = df_pairs.apply(get_tfidf_sim, axis=1)

# ========== 5. Calculate number of common authors ==========
def num_common_authors(row):
    authors1 = author_dict.get(row["refer_paper_id"], set())
    authors2 = author_dict.get(row["unique_paper_id"], set())
    return len(authors1.intersection(authors2))

df_pairs["num_common_author"] = df_pairs.apply(num_common_authors, axis=1)

# ========== 6. Calculate absolute difference in publication years ==========
def gap_year(row):
    y1 = year_dict.get(row["refer_paper_id"])
    y2 = year_dict.get(row["unique_paper_id"])
    try:
        return abs(int(y1) - int(y2))
    except (TypeError, ValueError):
        return None

df_pairs["gap_year"] = df_pairs.apply(gap_year, axis=1)

# ========== 7. Build citation graph as directed graph ==========
G = nx.DiGraph()
G.add_nodes_from(paper_ids)
for idx, refs in ref_dict.items():
    for ref in refs:
        if ref in paper_ids:  # Only add edge if referenced paper is in the dataset
            G.add_edge(idx, ref)

# Create reversed and undirected versions of the graph
G_rev = G.reverse()
G_undirected = G.to_undirected()

# ========== 8. Define helper functions to get neighbors within a certain depth ==========
def get_neighbors(graph, node, depth):
    try:
        # Returns set of nodes reachable within 'depth' hops from 'node', excluding 'node' itself
        return set(nx.single_source_shortest_path_length(graph, node, cutoff=depth).keys()) - {node}
    except Exception:
        return set()

def compute_common_neighbors(row, graph, depth):
    n1 = get_neighbors(graph, row["refer_paper_id"], depth)
    n2 = get_neighbors(graph, row["unique_paper_id"], depth)
    # Return the size of intersection of neighbors sets
    return len(n1 & n2)

# ========== 9. Calculate common neighbors features for different depths ==========
for depth in [1, 2, 3]:
    df_pairs[f"num_common_neighbors_lvl_{depth}"] = df_pairs.apply(
        lambda row: compute_common_neighbors(row, G, depth), axis=1)
    df_pairs[f"rev_num_common_neighbors_lvl_{depth}"] = df_pairs.apply(
        lambda row: compute_common_neighbors(row, G_rev, depth), axis=1)
    df_pairs[f"mix_num_common_neighbors_lvl_{depth}"] = df_pairs.apply(
        lambda row: compute_common_neighbors(row, G_undirected, depth), axis=1)

# ========== 10. Save the enriched dataset with features ==========
df_pairs.to_csv("paper_pairs_dataset_with_all_features.csv", index=False)
print("✅ File paper_pairs_dataset_with_all_features.csv has been created.")
