import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score



def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    precision = precision_score(
        torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy(),
        torch.cat([pos_score, neg_score]).numpy() > 0,
    )
    recall = recall_score(
        torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy(),
        torch.cat([pos_score, neg_score]).numpy() > 0,
    )
    accuracy = accuracy_score(
        torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy(),
        torch.cat([pos_score, neg_score]).numpy() > 0,
    )
    return roc_auc_score(labels, scores), precision, recall, accuracy
