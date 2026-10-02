"""Extract user preference features from review history."""
import pandas as pd
import numpy as np
from collections import Counter


class UserProfileBuilder:
    """Build user preference profiles from review behavior."""

    def __init__(self, reviews_df, businesses_df):
        self.reviews = reviews_df
        self.businesses = businesses_df
        self.profiles = {}

    def get_user_primary_city(self, user_id):
        """Get user's primary city (most reviews)."""
        user_reviews = self.reviews[self.reviews['user_id'] == user_id]
        merged = user_reviews.merge(
            self.businesses[['business_id', 'city']],
            on='business_id'
        )
        if len(merged) == 0:
            return None
        return merged['city'].value_counts().index[0]

    def get_user_cities(self, user_id):
        """Get all cities where user has reviewed."""
        user_reviews = self.reviews[self.reviews['user_id'] == user_id]
        merged = user_reviews.merge(
            self.businesses[['business_id', 'city']],
            on='business_id'
        )
        return merged['city'].unique().tolist()

    def get_user_rating_behavior(self, user_id):
        """Get user's rating distribution."""
        user_reviews = self.reviews[self.reviews['user_id'] == user_id]
        return {
            'mean_rating': user_reviews['stars'].mean(),
            'median_rating': user_reviews['stars'].median(),
            'std_rating': user_reviews['stars'].std(),
            'num_reviews': len(user_reviews)
        }

    def get_user_category_preferences(self, user_id, top_n=10):
        """Get user's preferred business categories."""
        user_reviews = self.reviews[self.reviews['user_id'] == user_id]
        merged = user_reviews.merge(
            self.businesses[['business_id', 'categories']],
            on='business_id'
        )

        categories = []
        for cats in merged['categories'].dropna():
            if isinstance(cats, list):
                categories.extend(cats)
            elif isinstance(cats, str):
                categories.extend([c.strip() for c in cats.split(',')])

        if not categories:
            return []

        return [cat for cat, _ in Counter(categories).most_common(top_n)]

    def get_average_price_level(self, user_id):
        """Get user's average price level preference."""
        if 'price' not in self.businesses.columns:
            return None

        user_reviews = self.reviews[self.reviews['user_id'] == user_id]
        merged = user_reviews.merge(
            self.businesses[['business_id', 'price']],
            on='business_id'
        )
        price_levels = merged['price'].dropna()
        if len(price_levels) > 0:
            return price_levels.mean()
        return None

    def build_profile(self, user_id):
        """Build complete user profile."""
        if user_id in self.profiles:
            return self.profiles[user_id]

        profile = {
            'user_id': user_id,
            'primary_city': self.get_user_primary_city(user_id),
            'cities_visited': self.get_user_cities(user_id),
            'rating_behavior': self.get_user_rating_behavior(user_id),
            'preferred_categories': self.get_user_category_preferences(user_id),
            'avg_price_level': self.get_average_price_level(user_id)
        }

        self.profiles[user_id] = profile
        return profile
