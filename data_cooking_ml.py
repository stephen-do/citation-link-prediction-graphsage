import pandas as pd
import networkx as nx
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1. Đọc và xử lý dữ liệu ==========
df = pd.read_csv("dataset/citation-cooked.csv")  # <-- Thay bằng file CSV của bạn
df = df.dropna(subset=["Index"])
df["Index"] = df["Index"].astype(int)
df = df.head(1000)

# Parse References
def parse_references(ref_str):
    if pd.isna(ref_str) or ref_str == "None":
        return []
    return [int(r.strip()) for r in str(ref_str).split(',') if r.strip().isdigit()]

df["References"] = df["References"].apply(parse_references)

# ========== 2. Tạo các cặp bài báo ==========
paper_ids = df["Index"].tolist()
pairs = [(i, j) for i, j in product(paper_ids, repeat=2) if i != j]
df_pairs = pd.DataFrame(pairs, columns=["refer_paper_id", "unique_paper_id"])

# ========== 3. Gắn nhãn forward_refer ==========
ref_dict = df.set_index("Index")["References"].to_dict()
df_pairs["forward_refer"] = df_pairs.apply(
    lambda row: int(row["refer_paper_id"] in ref_dict.get(row["unique_paper_id"], [])),
    axis=1,
)

# ========== 4. TF-IDF similarity ==========
titles = df.set_index("Index")["Title"].fillna("")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(titles)
similarity_matrix = cosine_similarity(tfidf_matrix)

df_pairs["tf_idf_similarity"] = df_pairs.apply(
    lambda row: similarity_matrix[row["refer_paper_id"], row["unique_paper_id"]],
    axis=1,
)

# ========== 5. Số tác giả chung ==========
author_dict = df.set_index("Index")["Authors"].fillna("").apply(
    lambda x: set([a.strip() for a in x.split(",") if a.strip()])
)
df_pairs["num_common_author"] = df_pairs.apply(
    lambda row: len(author_dict[row["refer_paper_id"]] & author_dict[row["unique_paper_id"]]),
    axis=1,
)

# ========== 6. Khoảng cách năm xuất bản ==========
year_dict = df.set_index("Index")["Year"]
df_pairs["gap_year"] = df_pairs.apply(
    lambda row: abs(int(year_dict.get(row["refer_paper_id"], 0)) - int(year_dict.get(row["unique_paper_id"], 0))),
    axis=1,
)

# ========== 7. Xây dựng mạng trích dẫn ==========
G = nx.DiGraph()
G.add_nodes_from(paper_ids)
for row in df.itertuples():
    for ref in row.References:
        G.add_edge(row.Index, ref)

G_rev = G.reverse()
G_undirected = G.to_undirected()

# ========== 8. Hàm tính common neighbors ==========
def get_neighbors(graph, node, depth):
    try:
        return set(nx.single_source_shortest_path_length(graph, node, cutoff=depth).keys()) - {node}
    except:
        return set()

def compute_common_neighbors(row, graph, depth):
    n1 = get_neighbors(graph, row["refer_paper_id"], depth)
    n2 = get_neighbors(graph, row["unique_paper_id"], depth)
    return len(n1 & n2)

# ========== 9. Tính tất cả các đặc trưng graph ==========
for depth in [1, 2, 3]:
    df_pairs[f"num_common_neighbors_lvl_{depth}"] = df_pairs.apply(
        lambda row: compute_common_neighbors(row, G, depth), axis=1)
    df_pairs[f"rev_num_common_neighbors_lvl_{depth}"] = df_pairs.apply(
        lambda row: compute_common_neighbors(row, G_rev, depth), axis=1)
    df_pairs[f"mix_num_common_neighbors_lvl_{depth}"] = df_pairs.apply(
        lambda row: compute_common_neighbors(row, G_undirected, depth), axis=1)

# ========== 10. Xuất kết quả ==========
df_pairs.to_csv("paper_pairs_dataset_with_all_features.csv", index=False)
print("✅ File paper_pairs_dataset_with_all_features.csv đã được tạo.")
