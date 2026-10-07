"""Baseline recommendation models for Phase 1."""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class PopularityBaseline:
    """Rank destinations by popularity (num reviews + rating)."""

    def __init__(self, businesses_df, reviews_df):
        self.businesses = businesses_df
        self.reviews = reviews_df

    def score_city(self, city):
        """Score a city by its popularity."""
        city_businesses = self.businesses[self.businesses['city'] == city]
        if len(city_businesses) == 0:
            return 0.0

        avg_rating = city_businesses['stars'].mean()
        num_reviews = city_businesses['review_count'].sum()

        # Normalize: popularity = avg_rating * log(num_reviews)
        popularity_score = avg_rating * np.log1p(num_reviews)
        return popularity_score

    def recommend(self, user_id, candidate_cities, k=10):
        """Rank candidate cities by popularity."""
        scores = {city: self.score_city(city) for city in candidate_cities}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [city for city, _ in ranked[:k]]


class CosineSimilarityBaseline:
    """Recommend cities similar to user's preferences."""

    def __init__(self, businesses_df, reviews_df):
        self.businesses = businesses_df
        self.reviews = reviews_df

        # Fit one scaler across all cities so users and cities share the same scale
        self.city_vectors = {}
        for city in businesses_df['city'].unique():
            city_vec = self.build_city_vector(city)
            self.city_vectors[city] = self._to_array(city_vec)
        self.scaler = StandardScaler().fit(np.vstack(list(self.city_vectors.values())))

    def _to_array(self, features):
        """Feature dict -> 1-row array in a fixed order."""
        return np.array([
            features['avg_rating'],
            features['avg_review_count'],
            np.log1p(features['num_reviews'])
        ]).reshape(1, -1)

    def build_user_vector(self, user_id):
        """Build feature vector for user from their reviews."""
        user_reviews = self.reviews[self.reviews['user_id'] == user_id]
        merged = user_reviews[['business_id']].merge(
            self.businesses[['business_id', 'stars', 'review_count']],
            on='business_id'
        )

        if len(merged) == 0:
            return None

        features = {
            'avg_rating': merged['stars'].mean(),
            'avg_review_count': np.log1p(merged['review_count'].mean()),
            'num_reviews': len(merged)
        }

        return features

    def build_city_vector(self, city):
        """Build feature vector for city."""
        city_businesses = self.businesses[self.businesses['city'] == city]
        if len(city_businesses) == 0:
            return None

        features = {
            'avg_rating': city_businesses['stars'].mean(),
            'avg_review_count': np.log1p(city_businesses['review_count'].mean()),
            'num_reviews': city_businesses['review_count'].sum()
        }

        return features

    def recommend(self, user_id, candidate_cities, k=10):
        """Rank cities by cosine similarity to user profile."""
        user_vec = self.build_user_vector(user_id)
        if user_vec is None:
            return []

        user_features_norm = self.scaler.transform(self._to_array(user_vec))

        scores = {}
        for city in candidate_cities:
            if city not in self.city_vectors:
                scores[city] = 0.0
                continue

            city_features_norm = self.scaler.transform(self.city_vectors[city])
            scores[city] = cosine_similarity(user_features_norm, city_features_norm)[0, 0]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [city for city, _ in ranked[:k]]
