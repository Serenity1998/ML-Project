"""Two-tower neural network for Phase 4."""
import torch
import torch.nn as nn


class UserTower(nn.Module):
    """Neural network to encode user features. (Phase 4)"""

    def __init__(self, input_dim, embedding_dim=64):
        """
        input_dim: number of input features
        embedding_dim: dimension of learned embedding
        """
        super().__init__()
        pass

    def forward(self, x):
        """Encode user features to embedding. (Phase 4)"""
        pass


class DestinationTower(nn.Module):
    """Neural network to encode destination features. (Phase 4)"""

    def __init__(self, input_dim, embedding_dim=64):
        """
        input_dim: number of input features
        embedding_dim: dimension of learned embedding
        """
        super().__init__()
        pass

    def forward(self, x):
        """Encode destination features to embedding. (Phase 4)"""
        pass


class TwoTowerRecommender(nn.Module):
    """Two-tower model for user-destination matching. (Phase 4)"""

    def __init__(self, user_input_dim, dest_input_dim, embedding_dim=64):
        super().__init__()
        self.user_tower = UserTower(user_input_dim, embedding_dim)
        self.dest_tower = DestinationTower(dest_input_dim, embedding_dim)

    def forward(self, user_features, dest_features):
        """Compute compatibility score. (Phase 4)"""
        pass

    def train_model(self, train_data, val_data, epochs=10, batch_size=32):
        """Train the model. (Phase 4)"""
        pass

    def recommend(self, user_features, candidate_dest_features, k=10):
        """Recommend top-K destinations. (Phase 4)"""
        pass
