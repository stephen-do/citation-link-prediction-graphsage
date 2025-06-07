import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def compute_loss(pos_score, neg_score):
    """
    Compute binary cross-entropy loss between positive and negative edge scores.

    Args:
        pos_score (Tensor): Predicted scores for positive edges (shape: [N]).
        neg_score (Tensor): Predicted scores for negative edges (shape: [M]).

    Returns:
        Tensor: Scalar loss value.
    """
    # Concatenate positive and negative scores
    scores = torch.cat([pos_score, neg_score])
    # Create labels: 1 for positive edges, 0 for negative
    labels = torch.cat([
        torch.ones(pos_score.shape[0], device=scores.device),
        torch.zeros(neg_score.shape[0], device=scores.device)
    ])
    # Use binary cross-entropy with logits
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    """
    Compute AUC, precision, recall, and accuracy for predicted edge scores.

    Args:
        pos_score (Tensor): Predicted scores for positive edges (logits).
        neg_score (Tensor): Predicted scores for negative edges (logits).

    Returns:
        tuple:
            - auc (float): Area under ROC curve.
            - precision (float): Precision score.
            - recall (float): Recall score.
            - accuracy (float): Accuracy score.
    """
    # Combine scores and labels
    scores = torch.cat([pos_score, neg_score]).detach().cpu()
    labels = torch.cat([
        torch.ones(pos_score.shape[0]),
        torch.zeros(neg_score.shape[0])
    ])

    # Convert to NumPy arrays
    scores_np = scores.numpy()
    labels_np = labels.numpy()

    # Convert logits to binary predictions (threshold = 0)
    preds_np = (scores_np > 0).astype(int)

    # Compute metrics
    auc = roc_auc_score(labels_np, scores_np)
    precision = precision_score(labels_np, preds_np)
    recall = recall_score(labels_np, preds_np)
    accuracy = accuracy_score(labels_np, preds_np)

    return auc, precision, recall, accuracy
