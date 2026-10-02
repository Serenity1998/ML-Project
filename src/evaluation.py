"""Evaluation metrics for recommender systems."""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def precision_at_k(recommendations, test_reviews, k=10):
    """Precision@K: fraction of recommendations that were actually reviewed."""
    if len(recommendations) == 0:
        return 0.0
    reviewed = test_reviews['business_id'].tolist()
    hits = sum(1 for rec in recommendations[:k] if rec in reviewed)
    return hits / min(k, len(recommendations))


def recall_at_k(recommendations, test_reviews, k=10):
    """Recall@K: fraction of test reviews present in top-K recommendations."""
    reviewed = test_reviews['business_id'].tolist()
    if len(reviewed) == 0:
        return 0.0
    hits = sum(1 for rec in recommendations[:k] if rec in reviewed)
    return hits / len(reviewed)


def ndcg_at_k(recommendations, test_reviews, k=10):
    """Normalized Discounted Cumulative Gain."""
    reviewed = test_reviews['business_id'].tolist()
    dcg = sum(1 / np.log2(i + 2) for i, rec in enumerate(recommendations[:k]) if rec in reviewed)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(reviewed))))
    return dcg / idcg if idcg > 0 else 0.0


def rating_mae(predicted_ratings, actual_ratings):
    """Mean absolute error of predicted ratings."""
    return mean_absolute_error(actual_ratings, predicted_ratings)


def evaluate_recommender(recommender, test_set, k=10):
    """Evaluate recommender on a test set."""
    metrics = {'precision': [], 'recall': [], 'ndcg': []}

    for user_id in test_set['user_id'].unique():
        user_test = test_set[test_set['user_id'] == user_id]
        recommendations = recommender.recommend(user_id, n_recommendations=k)

        if recommendations is not None:
            rec_ids = recommendations['business_id'].tolist()
            metrics['precision'].append(precision_at_k(rec_ids, user_test, k))
            metrics['recall'].append(recall_at_k(rec_ids, user_test, k))
            metrics['ndcg'].append(ndcg_at_k(rec_ids, user_test, k))

    return {
        'mean_precision': np.mean(metrics['precision']),
        'mean_recall': np.mean(metrics['recall']),
        'mean_ndcg': np.mean(metrics['ndcg']),
    }
