"""Feature engineering with budget/price level integration."""
import pandas as pd
import numpy as np


class FeatureEngineerWithBudget:
    """Build features for training, including price level."""

    def __init__(self):
        pass

    def build_features(self, reviews_df, users_df, businesses_df, weather_df=None, include_price=True):
        """Build complete feature set for model training.

        Args:
            reviews_df: Review records with user_id, business_id, rating
            users_df: User profiles
            businesses_df: Business data (must include 'price_level' column if include_price=True)
            weather_df: Optional weather features
            include_price: Whether to include price_level feature

        Returns:
            DataFrame with all features merged
        """
        # Start with reviews
        features = reviews_df.copy()

        # Merge user features
        user_features = self._extract_user_features(users_df)
        features = features.merge(user_features, on='user_id', how='left')

        # Merge business features
        business_features = self._extract_business_features(businesses_df, include_price=include_price)
        features = features.merge(business_features, on='business_id', how='left')

        # Merge weather if provided
        if weather_df is not None:
            features = self._merge_weather(features, weather_df)

        return features

    def _extract_user_features(self, users_df):
        """Extract user-level features."""
        user_features = users_df[[
            'user_id', 'review_count', 'useful', 'funny', 'cool',
            'average_stars', 'fans'
        ]].copy()

        # Normalize numeric features
        user_features['user_review_count_norm'] = user_features['review_count'] / users_df['review_count'].max()
        user_features['user_fans_norm'] = user_features['fans'] / users_df['fans'].max()

        # Rename for clarity
        user_features = user_features.rename(columns={
            'review_count': 'user_review_count',
            'average_stars': 'user_avg_rating',
            'fans': 'user_fans'
        })

        return user_features

    def _extract_business_features(self, businesses_df, include_price=True):
        """Extract business-level features."""
        business_features = businesses_df[[
            'business_id', 'name', 'city', 'stars', 'review_count'
        ]].copy()

        business_features = business_features.rename(columns={
            'stars': 'business_stars',
            'review_count': 'business_review_count'
        })

        # Add price level if available
        if include_price and 'price_level' in businesses_df.columns:
            business_features['price_level'] = businesses_df['price_level']

        return business_features

    def _merge_weather(self, features, weather_df):
        """Merge weather features based on city and month of review."""
        # Extract month from review date
        if 'date' in features.columns:
            features['review_month'] = pd.to_datetime(features['date']).dt.month
        else:
            # If no date column, default to month 1
            features['review_month'] = 1

        # Merge weather on city + month
        features = features.merge(
            weather_df,
            left_on=['city', 'review_month'],
            right_on=['city', 'month'],
            how='left'
        )

        return features

    def get_feature_set(self, features, feature_config):
        """Extract specific features based on config.

        Args:
            features: Full feature DataFrame
            feature_config: List of feature groups (e.g., ["user_rating_behavior", "weather"])

        Returns:
            DataFrame with selected features and target
        """
        selected_features = ['rating']  # Target variable

        for config in feature_config:
            if config == "user_rating_behavior":
                selected_features.extend([
                    'user_review_count', 'user_avg_rating', 'user_fans',
                    'user_review_count_norm', 'user_fans_norm'
                ])

            elif config == "destination_stats":
                selected_features.extend([
                    'business_stars', 'business_review_count'
                ])

            elif config == "price_level":
                selected_features.append('price_level')

            elif config == "weather":
                weather_cols = [col for col in features.columns if col.startswith('avg_') or col.startswith('total_')]
                selected_features.extend(weather_cols)

            elif config == "review_embeddings":
                embedding_cols = [col for col in features.columns if col.startswith('embedding_')]
                selected_features.extend(embedding_cols)

        # Return features that exist in the DataFrame
        existing_features = [f for f in selected_features if f in features.columns]

        return features[existing_features].dropna()


def create_training_data(reviews_df, users_df, businesses_df, weather_df=None, feature_config=None):
    """Convenience function to create training data with specific features.

    Args:
        reviews_df: Review records
        users_df: User profiles
        businesses_df: Business data with price_level
        weather_df: Optional weather features
        feature_config: Feature configuration (from config.FEATURE_CONFIGS)

    Returns:
        tuple: (X, y) where X is features and y is ratings
    """
    if feature_config is None:
        feature_config = ["user_rating_behavior", "destination_stats"]

    engineer = FeatureEngineerWithBudget()

    # Build all features
    all_features = engineer.build_features(
        reviews_df, users_df, businesses_df, weather_df,
        include_price=("price_level" in str(feature_config))
    )

    # Extract selected features
    feature_df = engineer.get_feature_set(all_features, feature_config)

    # Split X and y
    X = feature_df.drop('rating', axis=1)
    y = feature_df['rating']

    return X, y
