"""Evaluation harness for recommendation systems."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.evaluation.metrics import aggregate_metrics
from src.data.labels import (
    build_user_city_table,
    assign_target_cities,
    split_history,
    build_training_labels,
)


class RecommendationHarness:
    """Evaluation framework for destination recommendations."""

    def __init__(self, reviews_df, businesses_df, test_size=0.2, random_state=42, n_negatives=4, val_size=0.0):
        self.reviews = reviews_df
        self.businesses = businesses_df
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.n_negatives = n_negatives
        self.user_city = None
        self.targets = None
        self.train_users = None
        self.test_users = None
        self.val_users = []
        self.train_reviews = None
        self.test_reviews = None
        self.val_reviews = None
        self._visited = None

    def create_train_test_split(self):
        """
        Hold out one target city per user, then split users (not reviews) into train/test.
        Returned reviews are history only: each user's target-city reviews are removed.
        """
        self.user_city = build_user_city_table(self.reviews, self.businesses)
        self.targets = assign_target_cities(self.user_city, self.random_state)
        history = split_history(self.reviews, self.businesses, self.targets)

        users = sorted(self.targets['user_id'])
        self.train_users, self.test_users = train_test_split(
            users,
            test_size=self.test_size,
            random_state=self.random_state
        )
        # Validation users come out of train, so test users stay the same with or without them
        if self.val_size:
            self.train_users, self.val_users = train_test_split(
                self.train_users,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state
            )

        self.train_reviews = history[history['user_id'].isin(self.train_users)].copy()
        self.test_reviews = history[history['user_id'].isin(self.test_users)].copy()
        self.val_reviews = history[history['user_id'].isin(self.val_users)].copy()

        return self.train_reviews, self.test_reviews

    def get_training_labels(self, all_cities):
        """(user_id, city, label) rows for train users: target city = 1, sampled unvisited = 0."""
        train_targets = self.targets[self.targets['user_id'].isin(self.train_users)]
        return build_training_labels(
            self.user_city,
            all_cities,
            train_targets,
            self.n_negatives,
            self.random_state
        )

    def get_test_pairs(self):
        """(user, held_out_city) pairs for evaluation: one held-out city per test user."""
        test_targets = self.targets[self.targets['user_id'].isin(self.test_users)]
        return list(test_targets.itertuples(index=False, name=None))

    def get_validation_pairs(self):
        """(user, held_out_city) pairs for tuning: one held-out city per validation user."""
        val_targets = self.targets[self.targets['user_id'].isin(self.val_users)]
        return list(val_targets.itertuples(index=False, name=None))

    def get_ranking_candidates(self, user_id, held_out_city, all_cities):
        """
        Get cities to rank for (user, held_out_city).
        All cities except the user's history cities, so held_out_city competes with unvisited ones.
        """
        if self._visited is None:
            self._visited = self.user_city.groupby('user_id')['city'].apply(set).to_dict()
        history_cities = self._visited[user_id] - {held_out_city}
        return [city for city in all_cities if city not in history_cities]

    def evaluate_ranking(self, rankings, held_out_city, k=10):
        """
        Evaluate a ranking.
        rankings: list of cities ranked by recommendation score (highest first)
        held_out_city: the ground truth city
        k: cutoff for Recall@K, NDCG@K, MRR
        """
        # Ensure held_out_city is in rankings
        if held_out_city not in rankings:
            # Add at end with lowest score
            rankings = list(rankings) + [held_out_city]

        # Find position of held_out_city
        try:
            pos = rankings.index(held_out_city)
        except ValueError:
            pos = len(rankings)

        # Recall@K: is the city in top-K?
        recall_at_k = 1.0 if pos < k else 0.0

        # NDCG@K: discounted gain based on position
        if pos < k:
            ideal_position = 0  # Best possible ranking
            dcg = 1.0 / np.log2(pos + 2)
            idcg = 1.0 / np.log2(ideal_position + 2)
            ndcg_at_k = dcg / idcg
        else:
            ndcg_at_k = 0.0

        # MRR: reciprocal rank
        mrr = 1.0 / (pos + 1) if pos < k else 0.0

        return {
            'recall@k': recall_at_k,
            'ndcg@k': ndcg_at_k,
            'mrr': mrr,
            'position': pos
        }

    def evaluate(self, pairs, all_cities, score_fn, k=10):
        """
        Rank each user's candidate cities by score_fn and aggregate Recall/NDCG/MRR@k.
        score_fn: takes a DataFrame of (user_id, city) rows, returns one score per row.
        """
        rows = [
            (user_id, city)
            for user_id, held_out_city in pairs
            for city in self.get_ranking_candidates(user_id, held_out_city, all_cities)
        ]
        candidates = pd.DataFrame(rows, columns=['user_id', 'city'])
        candidates['score'] = score_fn(candidates)

        held_out = dict(pairs)
        results = []
        for user_id, group in candidates.groupby('user_id', sort=False):
            rankings = group.sort_values('score', ascending=False)['city'].tolist()
            results.append(self.evaluate_ranking(rankings, held_out[user_id], k))
        return aggregate_metrics(results, k)
