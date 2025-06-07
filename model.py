import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(nn.Module):
    """
    A 4-layer GraphSAGE model using mean aggregation.

    Args:
        in_feats (int): Input feature size for each node.
        h_feats (int): Hidden feature size (kept constant across all layers).
    """
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()

        # Define GraphSAGE layers
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type="mean")
        self.conv3 = SAGEConv(h_feats, h_feats, aggregator_type="mean")
        self.conv4 = SAGEConv(h_feats, h_feats, aggregator_type="mean")

    def forward(self, g, in_feat):
        """
        Forward pass through the 4-layer GraphSAGE model.

        Args:
            g (DGLGraph): The input DGL graph.
            in_feat (Tensor): Node feature tensor of shape [num_nodes, in_feats].

        Returns:
            Tensor: Output node representations of shape [num_nodes, h_feats].
        """
        h = self.conv1(g, in_feat)
        h = F.relu(h)

        h = self.conv2(g, h)
        h = F.relu(h)

        h = self.conv3(g, h)
        h = F.relu(h)

        h = self.conv4(g, h)

        return h
