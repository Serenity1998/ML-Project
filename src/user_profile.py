"""Build and manage user profiles for cross-city recommendations."""
import numpy as np
import pandas as pd


class UserProfile:
    """Represent a user's cross-city preferences and review history."""

    def __init__(self, user_id, reviews_df, businesses_df):
        self.user_id = user_id
        self.reviews = reviews_df[reviews_df['user_id'] == user_id]
        self.businesses = businesses_df
        self._build_profile()

    def _build_profile(self):
        """Build the user profile from review history."""
        merged = self.reviews.merge(
            self.businesses[['business_id', 'city', 'categories', 'stars']],
            on='business_id'
        )

        self.cities_visited = merged['city'].unique().tolist()
        self.avg_rating = merged['stars_x'].mean()
        self.preferred_categories = self._extract_categories(merged)
        self.city_ratings = merged.groupby('city')['stars_x'].mean().to_dict()

    def _extract_categories(self, merged_df):
        """Extract preferred business categories."""
        categories = []
        for cats in merged_df['categories'].dropna():
            if isinstance(cats, list):
                categories.extend(cats)
        return list(set(categories))

    def get_recommendation_cities(self, all_cities):
        """Get cities not yet visited by user."""
        return [c for c in all_cities if c not in self.cities_visited]
