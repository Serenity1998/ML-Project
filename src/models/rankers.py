"""Ranking models: score (user, city) rows by how likely the user visits the city."""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.models.two_tower import TwoTowerRecommender


class RandomForestRanker:
    """Random Forest classifier on visited (1) vs unvisited (0); its probability is the ranking score."""

    def __init__(self, features, n_estimators=100, max_depth=15, random_state=42):
        self.features = features
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, df):
        self.model.fit(df[self.features], df['label'])
        return self

    def score(self, df):
        return self.model.predict_proba(df[self.features])[:, 1]

    def feature_importance(self):
        return dict(zip(self.features, self.model.feature_importances_))


class TwoTowerRanker:
    """Two-Tower network (user tower · city tower) trained on visited (1) vs unvisited (0)."""

    def __init__(self, user_cols, city_cols, hidden_dim=128, embedding_dim=64,
                 learning_rate=0.001, batch_size=256, epochs=20, random_state=42):
        torch.manual_seed(random_state)
        self.user_cols = user_cols
        self.city_cols = city_cols
        self.batch_size = batch_size
        self.epochs = epochs
        self.user_scaler = StandardScaler()
        self.city_scaler = StandardScaler()
        self.model = TwoTowerRecommender(len(user_cols), len(city_cols), hidden_dim, embedding_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history = []

    def _tensors(self, df):
        users = torch.tensor(self.user_scaler.transform(df[self.user_cols].to_numpy()), dtype=torch.float32)
        cities = torch.tensor(self.city_scaler.transform(df[self.city_cols].to_numpy()), dtype=torch.float32)
        return users, cities

    def fit(self, df, val_score_fn=None):
        """Train with BCE; if val_score_fn is given, keep the epoch with the best validation score."""
        self.user_scaler.fit(df[self.user_cols].to_numpy())
        self.city_scaler.fit(df[self.city_cols].to_numpy())
        users, cities = self._tensors(df)
        labels = torch.tensor(df['label'].to_numpy(), dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss()

        best_score, best_state = -np.inf, None
        for _ in range(self.epochs):
            self.model.train()
            order = torch.randperm(len(labels))
            for start in range(0, len(labels), self.batch_size):
                idx = order[start:start + self.batch_size]
                loss = loss_fn(self.model(users[idx], cities[idx]), labels[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if val_score_fn is not None:
                val = val_score_fn(self)
                self.history.append(val)
                if val > best_score:
                    best_score, best_state = val, copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def score(self, df):
        self.model.eval()
        users, cities = self._tensors(df)
        with torch.no_grad():
            return self.model(users, cities).numpy()
