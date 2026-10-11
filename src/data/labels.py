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


def assign_target_cities(user_city_df, random_state=42, allowed=None):
    """
    Pick one held-out target city per user.
    allowed: optional {user_id: set of cities} the target must come from; users with none are dropped.
    """
    if allowed is not None:
        keep = [city in allowed.get(user, ()) for user, city in zip(user_city_df['user_id'], user_city_df['city'])]
        user_city_df = user_city_df[keep]
    targets = user_city_df.groupby('user_id').sample(n=1, random_state=random_state)
    return targets[['user_id', 'city']].sort_values('user_id').reset_index(drop=True)


def far_cities_by_user(user_city_df, city_coords, min_distance_km):
    """
    {user_id: cities at least min_distance_km from the user's home city (the one they reviewed most)}.
    city_coords: DataFrame indexed by city with 'lat' and 'lon' columns.
    """
    home = user_city_df.sort_values('num_reviews', ascending=False).drop_duplicates('user_id')
    home = home.set_index('user_id')['city']
    lat = np.radians(city_coords['lat'].to_numpy())
    lon = np.radians(city_coords['lon'].to_numpy())
    cities = city_coords.index.to_numpy()
    position = {city: i for i, city in enumerate(cities)}

    far = {}
    for user_id, home_city in home.items():
        i = position[home_city]
        a = np.sin((lat - lat[i]) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2
        far[user_id] = set(cities[6371 * 2 * np.arcsin(np.sqrt(a)) >= min_distance_km])
    return far


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


def sample_negatives(user_city_df, all_cities, targets_df, n_neg=4, random_state=42, allowed=None):
    """Sample cities each user never reviewed, labeled 0 (only from allowed[user_id] if given)."""
    rng = np.random.default_rng(random_state)
    visited = user_city_df.groupby('user_id')['city'].apply(set)
    cities = sorted(all_cities)

    rows = []
    for user_id in targets_df['user_id']:
        unvisited = [city for city in cities if city not in visited[user_id]
                     and (allowed is None or city in allowed[user_id])]
        chosen = rng.choice(unvisited, size=min(n_neg, len(unvisited)), replace=False)
        rows.extend((user_id, city, 0) for city in chosen)

    return pd.DataFrame(rows, columns=['user_id', 'city', 'label'])


def build_training_labels(user_city_df, all_cities, targets_df, n_neg=4, random_state=42, allowed=None):
    """Target city (label 1) plus sampled unvisited cities (label 0) for each user."""
    positives = targets_df[['user_id', 'city']].assign(label=1)
    negatives = sample_negatives(user_city_df, all_cities, targets_df, n_neg, random_state, allowed)
    return pd.concat([positives, negatives], ignore_index=True)
