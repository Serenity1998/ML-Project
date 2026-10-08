"""Random Forest model for Phase 4."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class RandomForestRecommender:
    """Random Forest for user-destination compatibility. (Phase 4)"""

    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        """Initialize Random Forest recommender."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance_ = None

    def build_feature_matrix(self, reviews_df, user_features_df, dest_features_df,
                            dest_embeddings, user_embeddings):
        """Build combined feature matrix for training.

        Returns:
            X: feature matrix (n_samples, n_features)
            y: target labels (ratings)
        """
        features_list = []
        targets = []

        for idx, row in reviews_df.iterrows():
            user_id = row['user_id']
            city = row['city']
            rating = row['rating']

            # Get user features - use defaults if user not in training set
            user_feat = user_features_df[user_features_df['user_id'] == user_id]
            if user_feat.empty:
                user_dict = {
                    'user_avg_rating': 3.5,
                    'user_rating_std': 1.0,
                    'user_num_reviews': 1.0,
                    'user_num_cities': 1.0
                }
            else:
                user_dict = {}
                for col in user_feat.columns:
                    if col != 'user_id':
                        user_dict[f'user_{col}'] = float(user_feat[col].values[0])

            # Get destination features
            dest_feat = dest_features_df[dest_features_df['city'] == city]
            if dest_feat.empty:
                continue

            dest_dict = {}
            for col in dest_feat.columns:
                if col != 'city':
                    val = dest_feat[col].values[0]
                    dest_dict[f'dest_{col}'] = float(val) if pd.notna(val) else 0.0

            # Combine features
            feature_vec = {**user_dict, **dest_dict}

            # User embedding (384D)
            if user_id in user_embeddings:
                emb = user_embeddings[user_id]
                for i, val in enumerate(emb):
                    feature_vec[f'user_emb_{i}'] = float(val)
            else:
                # Default embedding (zeros)
                for i in range(384):
                    feature_vec[f'user_emb_{i}'] = 0.0

            # Destination embedding (384D)
            if city in dest_embeddings:
                emb = dest_embeddings[city]
                for i, val in enumerate(emb):
                    feature_vec[f'dest_emb_{i}'] = float(val)
            else:
                # Default embedding (zeros)
                for i in range(384):
                    feature_vec[f'dest_emb_{i}'] = 0.0

            features_list.append(feature_vec)
            targets.append(rating)

        # Convert to DataFrame and ensure consistent column order
        X = pd.DataFrame(features_list).fillna(0)
        y = np.array(targets)

        # Store feature names if not set
        if self.feature_names is None:
            self.feature_names = sorted(X.columns.tolist())
            X = X[self.feature_names]
        else:
            # Ensure same columns as training
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_names]

        return X.values, y

    def train(self, X, y):
        """Train Random Forest on feature matrix.

        Args:
            X: feature matrix (n_samples, n_features)
            y: target labels (ratings)
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.feature_importance_ = self.model.feature_importances_

        return self

    def predict(self, X):
        """Predict ratings for feature matrix.

        Args:
            X: feature matrix (n_samples, n_features)

        Returns:
            predictions: predicted ratings
        """
        if len(X) == 0:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self, top_n=20):
        """Get top N important features.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance_ is None:
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)
