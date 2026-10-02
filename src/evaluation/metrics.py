"""Evaluation metrics for recommendations."""
import numpy as np


def recall_at_k(ground_truth, predictions, k=10):
    """Recall@K: is ground truth in top-K predictions?"""
    return 1.0 if ground_truth in predictions[:k] else 0.0


def ndcg_at_k(ground_truth, predictions, k=10):
    """NDCG@K: normalized discounted cumulative gain."""
    if ground_truth not in predictions:
        return 0.0

    pos = predictions.index(ground_truth)
    if pos >= k:
        return 0.0

    dcg = 1.0 / np.log2(pos + 2)
    idcg = 1.0  # Ideal: ground truth at position 0
    return dcg / idcg


def mean_reciprocal_rank(ground_truth, predictions, k=10):
    """MRR: reciprocal of rank position."""
    if ground_truth not in predictions:
        return 0.0

    pos = predictions.index(ground_truth)
    if pos >= k:
        return 0.0

    return 1.0 / (pos + 1)


def aggregate_metrics(results, k=10):
    """Aggregate metrics from multiple evaluations."""
    if not results:
        return {}

    recall_scores = [r.get('recall@k', 0) for r in results]
    ndcg_scores = [r.get('ndcg@k', 0) for r in results]
    mrr_scores = [r.get('mrr', 0) for r in results]

    return {
        f'recall@{k}': np.mean(recall_scores),
        f'ndcg@{k}': np.mean(ndcg_scores),
        f'mrr@{k}': np.mean(mrr_scores),
        'num_queries': len(results)
    }
