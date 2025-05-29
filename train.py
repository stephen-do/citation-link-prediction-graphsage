import itertools
import dgl.data
import numpy as np
import scipy.sparse as sp
import torch

from dgl.data.utils import load_graphs
from torch.utils.tensorboard import SummaryWriter
from model import GraphSAGE
from predictor import MLPPredictor
from utils import compute_loss, compute_auc

if __name__ == "__main__":
    graph_data = "dataset/citation.dgl"
    glist, label_dict = load_graphs(graph_data)  # glist will be [g1, g2]
    g = glist[0]
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # Create different graphs for training and testing
    train_g = dgl.remove_edges(g, eids[:test_size])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    h_feats = 128
    writer = SummaryWriter(log_dir=f"logs/{graph_data}_{h_feats}")
    model = GraphSAGE(train_g.ndata["feat"].shape[1], h_feats)
    # pred = DotPredictor()
    pred = MLPPredictor(h_feats)



    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), pred.parameters()), lr=0.001
    )
    n_epochs = 1000
    for e in range(n_epochs):
        h = model(train_g, train_g.ndata["feat"])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            print("In epoch {}, loss: {}".format(e, loss))
            with torch.no_grad():
                pos_score = pred(test_pos_g, h)
                neg_score = pred(test_neg_g, h)
                roc, precision, recall, accuracy = compute_auc(pos_score, neg_score)
                writer.add_scalar("Loss/train", loss, e)
                writer.add_scalar("ROC/train", roc, e)
                writer.add_scalar("Precision/train", precision, e)
                writer.add_scalar("Recall/train", recall, e)
                writer.add_scalar("Accuracy/train", accuracy, e)
