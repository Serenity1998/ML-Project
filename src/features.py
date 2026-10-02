"""Feature engineering for recommendations."""
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def extract_business_features(businesses_df):
    """Extract features from business data."""
    features = businesses_df[['business_id', 'city', 'categories', 'stars']].copy()
    features['stars_norm'] = MinMaxScaler().fit_transform(features[['stars']])
    return features


def extract_review_features(reviews_df):
    """Extract features from review data."""
    features = reviews_df[['user_id', 'business_id', 'stars', 'useful', 'funny', 'cool']].copy()
    features['engagement'] = features[['useful', 'funny', 'cool']].sum(axis=1)
    return features


def user_preferences(user_reviews_df, businesses_df):
    """Aggregate user preferences from reviews."""
    merged = user_reviews_df.merge(
        businesses_df[['business_id', 'categories', 'stars']],
        on='business_id'
    )

    preferences = {
        'avg_rating': merged['stars_x'].mean(),
        'avg_business_rating': merged['stars_y'].mean(),
    }

    return preferences
