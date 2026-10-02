"""Recommender systems for cross-city Yelp businesses."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class BaselineRecommender:
    """Simple baseline: recommend top-rated businesses in new cities."""

    def __init__(self, businesses_df, reviews_df):
        self.businesses = businesses_df
        self.reviews = reviews_df

    def recommend(self, user_id, target_city, n_recommendations=10):
        """Recommend top-rated businesses in target city not yet reviewed by user."""
        # Get user's reviewed businesses
        user_reviews = self.reviews[self.reviews['user_id'] == user_id]['business_id'].tolist()

        # Get businesses in target city
        city_businesses = self.businesses[self.businesses['city'] == target_city]

        # Filter out already reviewed
        candidates = city_businesses[~city_businesses['business_id'].isin(user_reviews)]

        # Sort by stars and return top N
        recommendations = candidates.nlargest(n_recommendations, 'stars')

        return recommendations[['business_id', 'name', 'stars', 'city']]


class ContentBasedRecommender:
    """Content-based recommender using business descriptions and categories."""

    def __init__(self, businesses_df, reviews_df, model_name='all-MiniLM-L6-v2'):
        self.businesses = businesses_df
        self.reviews = reviews_df
        self.model = SentenceTransformer(model_name)
        self.business_embeddings = None

    def fit(self):
        """Compute embeddings for all businesses."""
        # Use categories as proxy for content (review text if available)
        texts = self.businesses['categories'].fillna('').astype(str).tolist()
        self.business_embeddings = self.model.encode(texts)

    def recommend(self, user_id, target_city, n_recommendations=10):
        """Recommend businesses similar to user's preferences in target city."""
        if self.business_embeddings is None:
            self.fit()

        # Get user's reviewed businesses
        user_reviews = self.reviews[self.reviews['user_id'] == user_id]
        reviewed_business_ids = user_reviews['business_id'].tolist()
        reviewed_businesses = self.businesses[self.businesses['business_id'].isin(reviewed_business_ids)]

        # Average embedding of reviewed businesses
        reviewed_indices = reviewed_businesses.index.tolist()
        if reviewed_indices:
            user_embedding = self.business_embeddings[reviewed_indices].mean(axis=0)
        else:
            return None

        # Get candidates in target city
        city_businesses = self.businesses[self.businesses['city'] == target_city]
        candidates = city_businesses[~city_businesses['business_id'].isin(reviewed_business_ids)]
        candidate_indices = candidates.index.tolist()

        if not candidate_indices:
            return None

        candidate_embeddings = self.business_embeddings[candidate_indices]

        # Compute similarity and rank
        similarities = cosine_similarity([user_embedding], candidate_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]

        recommendations = candidates.iloc[top_indices][['business_id', 'name', 'stars', 'city']]

        return recommendations
