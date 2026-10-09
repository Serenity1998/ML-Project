"""
Ensemble Model Training Pipeline
Implements: Random Forest + XGBoost + Neural Network + LightGBM
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

class EnsembleRecommender:
    """
    Ensemble model combining multiple algorithms:
    - Random Forest
    - XGBoost
    - LightGBM
    - Neural Network (via scikit-learn MLPRegressor)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()

        # Initialize all models
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1
        )

        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )

        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )

        # Neural Network (MLP)
        from sklearn.neural_network import MLPRegressor
        self.nn_model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )

        # Model weights for ensemble (learned or manual)
        self.model_weights = {
            'rf': 0.25,
            'xgb': 0.25,
            'lgb': 0.25,
            'nn': 0.25
        }

        self.feature_names = None
        self.models_trained = False

    def fit(self, X, y):
        """
        Train all models on the data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        print("[Ensemble] Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest
        print("[RF] Training Random Forest...")
        self.rf_model.fit(X_scaled, y)
        rf_train_score = self.rf_model.score(X_scaled, y)
        print(f"    Random Forest R² (train): {rf_train_score:.4f}")

        # Train XGBoost
        print("[XGB] Training XGBoost...")
        self.xgb_model.fit(X_scaled, y)
        xgb_train_score = self.xgb_model.score(X_scaled, y)
        print(f"    XGBoost R² (train): {xgb_train_score:.4f}")

        # Train LightGBM
        print("[LGB] Training LightGBM...")
        self.lgb_model.fit(X_scaled, y)
        lgb_train_score = self.lgb_model.score(X_scaled, y)
        print(f"    LightGBM R² (train): {lgb_train_score:.4f}")

        # Train Neural Network
        print("[NN] Training Neural Network...")
        self.nn_model.fit(X_scaled, y)
        nn_train_score = self.nn_model.score(X_scaled, y)
        print(f"    Neural Network R² (train): {nn_train_score:.4f}")

        self.models_trained = True
        self.feature_names = getattr(X, 'columns', None)

        # Print ensemble info
        print("\n[Ensemble] All models trained!")
        print(f"    Ensemble weights: {self.model_weights}")

    def predict(self, X):
        """
        Make predictions using ensemble (weighted average).

        Args:
            X: Feature matrix

        Returns:
            Ensemble predictions
        """
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        # Get predictions from each model
        rf_pred = self.rf_model.predict(X_scaled)
        xgb_pred = self.xgb_model.predict(X_scaled)
        lgb_pred = self.lgb_model.predict(X_scaled)
        nn_pred = self.nn_model.predict(X_scaled)

        # Weighted ensemble prediction
        ensemble_pred = (
            self.model_weights['rf'] * rf_pred +
            self.model_weights['xgb'] * xgb_pred +
            self.model_weights['lgb'] * lgb_pred +
            self.model_weights['nn'] * nn_pred
        )

        return ensemble_pred

    def predict_with_confidence(self, X):
        """
        Get predictions with confidence intervals from all models.

        Args:
            X: Feature matrix

        Returns:
            Dict with ensemble pred, individual preds, and confidence
        """
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        # Get predictions from each model
        rf_pred = self.rf_model.predict(X_scaled)
        xgb_pred = self.xgb_model.predict(X_scaled)
        lgb_pred = self.lgb_model.predict(X_scaled)
        nn_pred = self.nn_model.predict(X_scaled)

        # Ensemble prediction
        ensemble_pred = (
            self.model_weights['rf'] * rf_pred +
            self.model_weights['xgb'] * xgb_pred +
            self.model_weights['lgb'] * lgb_pred +
            self.model_weights['nn'] * nn_pred
        )

        # Calculate confidence (standard deviation across models)
        all_preds = np.array([rf_pred, xgb_pred, lgb_pred, nn_pred])
        confidence = np.std(all_preds, axis=0)  # Lower std = higher confidence

        return {
            'ensemble': ensemble_pred,
            'rf': rf_pred,
            'xgb': xgb_pred,
            'lgb': lgb_pred,
            'nn': nn_pred,
            'confidence': confidence,
            'std_dev': np.std(all_preds, axis=0)
        }

    def get_feature_importance(self):
        """Get feature importance from tree-based models."""
        importance_dict = {
            'random_forest': self.rf_model.feature_importances_,
            'xgboost': self.xgb_model.feature_importances_,
            'lightgbm': self.lgb_model.feature_importances_
        }
        return importance_dict

    def evaluate(self, X, y):
        """Evaluate all models on test data."""
        X_scaled = self.scaler.transform(X)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rf_pred = self.rf_model.predict(X_scaled)
        xgb_pred = self.xgb_model.predict(X_scaled)
        lgb_pred = self.lgb_model.predict(X_scaled)
        nn_pred = self.nn_model.predict(X_scaled)
        ensemble_pred = self.predict(X)

        results = {}

        for name, pred in [('RF', rf_pred), ('XGB', xgb_pred),
                          ('LGB', lgb_pred), ('NN', nn_pred), ('Ensemble', ensemble_pred)]:
            mse = mean_squared_error(y, pred)
            mae = mean_absolute_error(y, pred)
            r2 = r2_score(y, pred)
            results[name] = {'MSE': mse, 'MAE': mae, 'R²': r2}

        return pd.DataFrame(results).T

    def save(self, path):
        """Save all models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, path / 'scaler.joblib')
        joblib.dump(self.rf_model, path / 'rf_model.joblib')
        joblib.dump(self.xgb_model, path / 'xgb_model.joblib')
        joblib.dump(self.lgb_model, path / 'lgb_model.joblib')
        joblib.dump(self.nn_model, path / 'nn_model.joblib')
        joblib.dump(self.model_weights, path / 'weights.joblib')

        print(f"[OK] Ensemble models saved to {path}")

    def load(self, path):
        """Load all models from disk."""
        path = Path(path)

        self.scaler = joblib.load(path / 'scaler.joblib')
        self.rf_model = joblib.load(path / 'rf_model.joblib')
        self.xgb_model = joblib.load(path / 'xgb_model.joblib')
        self.lgb_model = joblib.load(path / 'lgb_model.joblib')
        self.nn_model = joblib.load(path / 'nn_model.joblib')
        self.model_weights = joblib.load(path / 'weights.joblib')

        self.models_trained = True
        print(f"[OK] Ensemble models loaded from {path}")


# Example usage
if __name__ == "__main__":
    print("Travel Recommendation Ensemble Model Training\n")

    # Create dummy data
    X = np.random.randn(100, 50)
    y = np.random.randn(100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train ensemble
    ensemble = EnsembleRecommender()
    ensemble.fit(X_train, y_train)

    # Evaluate
    print("\nModel Performance (Test Set):")
    print(ensemble.evaluate(X_test, y_test))

    # Get predictions with confidence
    results = ensemble.predict_with_confidence(X_test[:5])
    print("\nSample Predictions (first 5):")
    for key in ['ensemble', 'rf', 'xgb', 'lgb', 'nn']:
        print(f"  {key}: {results[key][:3]}")
