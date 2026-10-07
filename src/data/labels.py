"""Construct user-destination labels with a leave-one-city-out holdout."""
import numpy as np
import pandas as pd


def build_user_city_table(reviews_df, businesses_df):
    """One row per (user, city) with review count and average rating."""
    merged = reviews_df.merge(businesses_df[['business_id', 'city']], on='business_id')
    return merged.groupby(['user_id', 'city']).agg(
        num_reviews=('stars', 'size'),
        avg_stars=('stars', 'mean')
    ).reset_index()


def assign_target_cities(user_city_df, random_state=42):
    """Pick one held-out target city per user."""
    targets = user_city_df.groupby('user_id').sample(n=1, random_state=random_state)
    return targets[['user_id', 'city']].sort_values('user_id').reset_index(drop=True)


def split_history(reviews_df, businesses_df, targets_df):
    """Drop each user's target-city reviews, leaving only their history."""
    merged = reviews_df.merge(businesses_df[['business_id', 'city']], on='business_id')
    merged = merged.merge(
        targets_df.rename(columns={'city': 'target_city'}),
        on='user_id',
        how='left'
    )
    history = merged[merged['city'] != merged['target_city']]
    return history[reviews_df.columns].reset_index(drop=True)


def sample_negatives(user_city_df, all_cities, targets_df, n_neg=4, random_state=42):
    """Sample cities each user never reviewed, labeled 0."""
    rng = np.random.default_rng(random_state)
    visited = user_city_df.groupby('user_id')['city'].apply(set)
    cities = sorted(all_cities)

    rows = []
    for user_id in targets_df['user_id']:
        unvisited = [city for city in cities if city not in visited[user_id]]
        chosen = rng.choice(unvisited, size=min(n_neg, len(unvisited)), replace=False)
        rows.extend((user_id, city, 0) for city in chosen)

    return pd.DataFrame(rows, columns=['user_id', 'city', 'label'])


def build_training_labels(user_city_df, all_cities, targets_df, n_neg=4, random_state=42):
    """Target city (label 1) plus sampled unvisited cities (label 0) for each user."""
    positives = targets_df[['user_id', 'city']].assign(label=1)
    negatives = sample_negatives(user_city_df, all_cities, targets_df, n_neg, random_state)
    return pd.concat([positives, negatives], ignore_index=True)
