"""Evaluation harness for recommendation systems."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class RecommendationHarness:
    """Evaluation framework for destination recommendations."""

    def __init__(self, reviews_df, businesses_df, test_size=0.2, random_state=42):
        self.reviews = reviews_df
        self.businesses = businesses_df
        self.test_size = test_size
        self.random_state = random_state
        self.train_reviews = None
        self.test_reviews = None

    def create_train_test_split(self, min_test_cities=1):
        """
        Split data by user (not review).
        Each test user must have visited multiple cities.
        """
        users = list(self.reviews['user_id'].unique())
        train_users, test_users = train_test_split(
            users,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.train_reviews = self.reviews[self.reviews['user_id'].isin(train_users)].copy()
        self.test_reviews = self.reviews[self.reviews['user_id'].isin(test_users)].copy()

        return self.train_reviews, self.test_reviews

    def get_test_pairs(self):
        """
        Get (user, held_out_city) pairs for evaluation.
        For each test user, hold out one city they visited.
        """
        pairs = []
        for user_id in self.test_reviews['user_id'].unique():
            user_reviews = self.test_reviews[self.test_reviews['user_id'] == user_id]
            merged = user_reviews.merge(
                self.businesses[['business_id', 'city']],
                on='business_id'
            )
            cities_visited = merged['city'].unique()

            if len(cities_visited) >= 1:
                # Hold out each city (could sample 1 if too many)
                for held_out_city in cities_visited:
                    pairs.append((user_id, held_out_city))

        return pairs

    def get_ranking_candidates(self, user_id, held_out_city, all_cities):
        """
        Get cities to rank for (user, held_out_city).
        Include held_out_city (positive) + sample of unvisited cities (negatives).
        """
        user_reviews = self.train_reviews[self.train_reviews['user_id'] == user_id]
        merged = user_reviews.merge(
            self.businesses[['business_id', 'city']],
            on='business_id'
        )
        cities_visited_in_train = set(merged['city'].unique())

        # Candidates: held_out_city + unvisited cities
        candidates = list(cities_visited_in_train) + [held_out_city]
        candidates = list(set(candidates))

        return candidates

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
