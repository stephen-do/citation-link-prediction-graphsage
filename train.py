import itertools
import dgl
import numpy as np
import scipy.sparse as sp
import torch

from dgl.data.utils import load_graphs
from torch.utils.tensorboard import SummaryWriter

from model import GraphSAGE
from predictor import MLPPredictor  # hoặc DotPredictor
from utils import compute_loss, compute_auc


def create_negative_edges(g, u, v):
    """
    Generate negative edges (non-existent links) by taking the complement of the graph's adjacency matrix.
    """
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    return neg_u, neg_v


def split_edges(u, v, g, test_frac=0.1):
    """
    Split edges into positive training/testing sets and generate corresponding negative edges.
    """
    eids = np.random.permutation(g.number_of_edges())
    test_size = int(len(eids) * test_frac)

    # Positive edge split
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Negative edges
    neg_u, neg_v = create_negative_edges(g, u, v)
    neg_indices = np.random.choice(len(neg_u), g.number_of_edges(), replace=False)

    test_neg_u, test_neg_v = neg_u[neg_indices[:test_size]], neg_v[neg_indices[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_indices[test_size:]], neg_v[neg_indices[test_size:]]

    return (test_pos_u, test_pos_v, train_pos_u, train_pos_v,
            test_neg_u, test_neg_v, train_neg_u, train_neg_v, eids)


if __name__ == "__main__":
    # Load preprocessed graph
    graph_path = "dataset/citation.dgl"
    glist, _ = load_graphs(graph_path)
    g = glist[0]
    u, v = g.edges()

    # Split positive/negative edges
    (test_pos_u, test_pos_v, train_pos_u, train_pos_v,
     test_neg_u, test_neg_v, train_neg_u, train_neg_v, eids) = split_edges(u, v, g)

    # Create subgraphs for training and evaluation
    train_g = dgl.remove_edges(g, eids[:len(test_pos_u)])  # remove test edges
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    # Initialize model and predictor
    in_feats = train_g.ndata["feat"].shape[1]
    h_feats = 128
    model = GraphSAGE(in_feats, h_feats)
    predictor = MLPPredictor(h_feats)  # or DotPredictor()

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=0.001)

    # Logging
    writer = SummaryWriter(log_dir=f"logs/{graph_path.replace('/', '_')}_{h_feats}")

    # Training loop
    n_epochs = 1000
    for epoch in range(n_epochs):
        model.train()
        h = model(train_g, train_g.ndata["feat"])
        pos_score = predictor(train_pos_g, h)
        neg_score = predictor(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation every 50 epochs
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                h = model(train_g, train_g.ndata["feat"])  # recompute node embeddings
                pos_score = predictor(test_pos_g, h)
                neg_score = predictor(test_neg_g, h)
                roc, precision, recall, accuracy = compute_auc(pos_score, neg_score)

                print(f"[Epoch {epoch}] Loss: {loss:.4f}, ROC AUC: {roc:.4f}, "
                      f"Precision: {precision:.4f}, Recall: {recall:.4f}, Acc: {accuracy:.4f}")

                # Log to TensorBoard
                writer.add_scalar("Loss/train", loss.item(), epoch)
                writer.add_scalar("ROC_AUC/test", roc, epoch)
                writer.add_scalar("Precision/test", precision, epoch)
                writer.add_scalar("Recall/test", recall, epoch)
                writer.add_scalar("Accuracy/test", accuracy, epoch)
