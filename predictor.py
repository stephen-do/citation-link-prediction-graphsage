import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class DotPredictor(nn.Module):
    """
    Predicts edge (link) scores by computing the dot product
    between source and destination node embeddings.

    Usage:
        predictor = DotPredictor()
        scores = predictor(graph, node_embeddings)
    """

    def forward(self, g, h):
        """
        Compute dot-product scores for edges.

        Args:
            g (dgl.DGLGraph): Input graph with edges.
            h (Tensor): Node embeddings of shape [num_nodes, hidden_dim].

        Returns:
            Tensor: Edge scores of shape [num_edges].
        """
        with g.local_scope():
            g.ndata["h"] = h
            # Compute dot product between source and destination node embeddings
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"].squeeze(1)  # Remove extra dimension


class MLPPredictor(nn.Module):
    """
    Predicts edge (link) scores using an MLP over concatenated
    node embeddings of edge pairs (source, destination).

    Usage:
        predictor = MLPPredictor(hidden_dim)
        scores = predictor(graph, node_embeddings)
    """

    def __init__(self, h_feats):
        """
        Args:
            h_feats (int): Hidden dimension size of node embeddings.
        """
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Compute score for each edge using an MLP.

        Args:
            edges: An object with edges.src and edges.dst containing node features.

        Returns:
            dict: Dictionary with key "score" containing edge scores.
        """
        h_src = edges.src["h"]
        h_dst = edges.dst["h"]
        h_cat = torch.cat([h_src, h_dst], dim=1)  # [num_edges, 2*h_feats]
        score = self.W2(F.relu(self.W1(h_cat))).squeeze(1)  # [num_edges]
        return {"score": score}

    def forward(self, g, h):
        """
        Compute edge scores via MLP using node embeddings.

        Args:
            g (dgl.DGLGraph): Input graph.
            h (Tensor): Node embeddings of shape [num_nodes, hidden_dim].

        Returns:
            Tensor: Edge scores of shape [num_edges].
        """
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]
