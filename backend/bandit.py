import numpy as np
import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "app.db"

class LinTSBandit:
    """
    Linear Thompson Sampling contextual bandit.
    Learns a linear model theta over contexts [rf_score, climate_match, activity_sim, popularity].
    Reward is 1 for thumbs up, 0 for thumbs down.
    Maintains shared state across users via SQLite.
    """

    def __init__(self, n_arms=39, n_features=4, alpha=1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha  # regularization / exploration parameter

        # Initialize V (precision matrix) and b (score vector)
        # V starts as lambda*I where lambda is regularization strength
        self.V = np.eye(n_features)
        self.b = np.zeros(n_features)

        self._load_state()

    def _load_state(self):
        """Load bandit state from SQLite."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT V, b FROM bandit_state WHERE id=1")
            row = cursor.fetchone()
            if row:
                self.V = np.array(json.loads(row[0]))
                self.b = np.array(json.loads(row[1]))
            conn.close()
        except:
            pass  # If DB doesn't exist, start from scratch

    def _save_state(self):
        """Save bandit state to SQLite."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bandit_state (
                id INTEGER PRIMARY KEY,
                V TEXT NOT NULL,
                b TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "INSERT OR REPLACE INTO bandit_state (id, V, b) VALUES (1, ?, ?)",
            (json.dumps(self.V.tolist()), json.dumps(self.b.tolist()))
        )
        conn.commit()
        conn.close()

    def select_arm(self, contexts):
        """
        Thompson sampling: sample theta ~ N(theta_hat, alpha*V^-1), select arm with highest expected reward.

        Args:
            contexts: np.array of shape (n_arms, n_features) where each row is context for one arm.

        Returns:
            arm_idx: integer index of selected arm (city)
        """
        # Compute posterior mean
        V_inv = np.linalg.inv(self.V + 1e-8 * np.eye(self.n_features))
        theta_hat = V_inv @ self.b

        # Sample theta
        theta = np.random.multivariate_normal(theta_hat, self.alpha * V_inv)

        # Compute expected reward for each arm
        scores = contexts @ theta
        return int(np.argmax(scores))

    def update(self, context, reward):
        """
        Update bandit state with feedback.

        Args:
            context: np.array of shape (n_features,) for the selected arm
            reward: 0 or 1
        """
        # Sherman-Morrison update for V and b
        context = np.atleast_1d(context).flatten()
        self.V += np.outer(context, context)
        self.b += reward * context
        self._save_state()

    def get_scores(self, contexts):
        """
        Get expected scores for all arms (for display/ranking).

        Args:
            contexts: np.array of shape (n_arms, n_features)

        Returns:
            np.array of shape (n_arms,) with expected scores
        """
        V_inv = np.linalg.inv(self.V + 1e-8 * np.eye(self.n_features))
        theta = V_inv @ self.b
        return contexts @ theta

def get_or_create_bandit():
    """Lazy-init bandit."""
    return LinTSBandit(n_arms=39, n_features=4)
