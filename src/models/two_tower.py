"""Two-tower neural network for Phase 4b."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class UserTower(nn.Module):
    """Neural network to encode user features."""

    def __init__(self, input_dim, hidden_dim=128, embedding_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )

    def forward(self, x):
        """Encode user features to embedding."""
        return self.fc(x)


class DestinationTower(nn.Module):
    """Neural network to encode destination features."""

    def __init__(self, input_dim, hidden_dim=128, embedding_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )

    def forward(self, x):
        """Encode destination features to embedding."""
        return self.fc(x)


class TwoTowerRecommender(nn.Module):
    """Two-tower model for user-destination matching."""

    def __init__(self, user_input_dim, dest_input_dim, hidden_dim=128, embedding_dim=64):
        super().__init__()
        self.user_tower = UserTower(user_input_dim, hidden_dim, embedding_dim)
        self.dest_tower = DestinationTower(dest_input_dim, hidden_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, user_features, dest_features):
        """Compute compatibility score via dot product."""
        user_emb = self.user_tower(user_features)
        dest_emb = self.dest_tower(dest_features)
        scores = torch.sum(user_emb * dest_emb, dim=1)
        return scores

    def get_user_embedding(self, user_features):
        """Get user embedding."""
        return self.user_tower(user_features)

    def get_dest_embedding(self, dest_features):
        """Get destination embedding."""
        return self.dest_tower(dest_features)


class RecommendationDataset(Dataset):
    """PyTorch dataset for recommendation task."""

    def __init__(self, user_features, dest_features, ratings):
        self.user_features = torch.FloatTensor(user_features)
        self.dest_features = torch.FloatTensor(dest_features)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user_features': self.user_features[idx],
            'dest_features': self.dest_features[idx],
            'rating': self.ratings[idx]
        }


class TwoTowerTrainer:
    """Trainer for Two-Tower model."""

    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, train_loader):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            user_feats = batch['user_features'].to(self.device)
            dest_feats = batch['dest_features'].to(self.device)
            ratings = batch['rating'].to(self.device)

            scores = self.model(user_feats, dest_feats)
            loss = self.criterion(scores, ratings)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.history['train_loss'].append(avg_loss)
        return avg_loss

    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                user_feats = batch['user_features'].to(self.device)
                dest_feats = batch['dest_features'].to(self.device)
                ratings = batch['rating'].to(self.device)

                scores = self.model(user_feats, dest_feats)
                loss = self.criterion(scores, ratings)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.history['val_loss'].append(avg_loss)
        return avg_loss

    def train(self, train_loader, val_loader, epochs=20, early_stopping_patience=5):
        """Train model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return self.history

    def predict(self, test_loader):
        """Get predictions on test set."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                user_feats = batch['user_features'].to(self.device)
                dest_feats = batch['dest_features'].to(self.device)

                scores = self.model(user_feats, dest_feats)
                predictions.extend(scores.cpu().numpy())

        return np.array(predictions)
