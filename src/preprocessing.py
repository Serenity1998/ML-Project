"""Data preprocessing and filtering."""
import pandas as pd
from config import MIN_USER_REVIEWS, MIN_USER_CITIES


def filter_multi_city_users(reviews_df, users_df, businesses_df, min_cities=MIN_USER_CITIES, min_reviews=MIN_USER_REVIEWS):
    """Filter for users who have reviewed businesses in multiple cities."""
    # Merge reviews with business cities
    merged = reviews_df.merge(
        businesses_df[['business_id', 'city']],
        on='business_id'
    )

    # Count unique cities per user
    user_city_counts = merged.groupby('user_id')['city'].nunique()
    multi_city_users = user_city_counts[user_city_counts >= min_cities].index.tolist()

    # Filter reviews and users
    filtered_reviews = reviews_df[reviews_df['user_id'].isin(multi_city_users)]

    # Further filter: min reviews per user
    user_review_counts = filtered_reviews['user_id'].value_counts()
    active_users = user_review_counts[user_review_counts >= min_reviews].index.tolist()

    return filtered_reviews[filtered_reviews['user_id'].isin(active_users)]


def clean_text(text):
    """Basic text cleaning."""
    if pd.isna(text):
        return ""
    return str(text).strip()
