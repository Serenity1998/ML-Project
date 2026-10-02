"""Extract destination (city) features from business data."""
import pandas as pd
import numpy as np


class DestinationProfileBuilder:
    """Build destination profiles from business data."""

    def __init__(self, businesses_df, reviews_df):
        self.businesses = businesses_df
        self.reviews = reviews_df
        self.profiles = {}

    def get_city_stats(self, city):
        """Get basic statistics for a city."""
        city_businesses = self.businesses[self.businesses['city'] == city]
        if len(city_businesses) == 0:
            return None

        return {
            'num_businesses': len(city_businesses),
            'avg_rating': city_businesses['stars'].mean(),
            'median_rating': city_businesses['stars'].median(),
            'avg_review_count': city_businesses['review_count'].mean(),
            'num_reviews': city_businesses['review_count'].sum()
        }

    def get_category_distribution(self, city, top_n=15):
        """Get most common business categories in city."""
        city_businesses = self.businesses[self.businesses['city'] == city]
        categories = []

        for cats in city_businesses['categories'].dropna():
            if isinstance(cats, list):
                categories.extend(cats)
            elif isinstance(cats, str):
                categories.extend([c.strip() for c in cats.split(',')])

        if not categories:
            return []

        from collections import Counter
        return [cat for cat, _ in Counter(categories).most_common(top_n)]

    def get_price_distribution(self, city):
        """Get price level distribution for city."""
        if 'price' not in self.businesses.columns:
            return None

        city_businesses = self.businesses[self.businesses['city'] == city]
        prices = city_businesses['price'].dropna()
        if len(prices) > 0:
            return {
                'avg_price': prices.mean(),
                'median_price': prices.median(),
                'price_dist': prices.value_counts().to_dict()
            }
        return None

    def get_average_review_sentiment(self, city):
        """Get average rating for reviews in city."""
        city_businesses = self.businesses[self.businesses['city'] == city]['business_id']
        city_reviews = self.reviews[self.reviews['business_id'].isin(city_businesses)]
        if len(city_reviews) > 0:
            return city_reviews['stars'].mean()
        return None

    def build_profile(self, city):
        """Build complete destination profile."""
        if city in self.profiles:
            return self.profiles[city]

        profile = {
            'city': city,
            'stats': self.get_city_stats(city),
            'categories': self.get_category_distribution(city),
            'price_info': self.get_price_distribution(city),
            'avg_review_sentiment': self.get_average_review_sentiment(city)
        }

        self.profiles[city] = profile
        return profile
