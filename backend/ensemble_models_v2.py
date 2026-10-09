"""
Enhanced Ensemble Model Training Pipeline with Preprocessing
Handles: Categorical encoding + Numerical scaling + Feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path

class EnsembleRecommender:
    """
    Ensemble model with proper preprocessing pipeline.

    Handles:
    - Numerical features (scaling)
    - Categorical features (one-hot encoding)
    - Text embeddings (passthrough)
    """

    def __init__(self, numerical_features=None, categorical_features=None, random_state=42):
        self.random_state = random_state
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []

        # Build preprocessing pipeline
        self.preprocessor = self._build_preprocessor()

        # Initialize models
        self.rf_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.nn_model = None

        self.model_weights = {
            'rf': 0.25,
            'xgb': 0.25,
            'lgb': 0.25,
            'nn': 0.25
        }

        self.models_trained = False

    def _build_preprocessor(self):
        """
        Build preprocessing pipeline:
        - Numerical: StandardScaler
        - Categorical: OneHotEncoder
        """
        transformers = []

        # Numerical features: scale
        if self.numerical_features:
            transformers.append(
                ('num', StandardScaler(), self.numerical_features)
            )

        # Categorical features: one-hot encode
        if self.categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 self.categorical_features)
            )

        if transformers:
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'  # Keep other columns as-is (e.g., embeddings)
            )
        else:
            # Fallback: just scale everything
            preprocessor = StandardScaler()

        return preprocessor

    def fit(self, X, y):
        """
        Train all models with preprocessing pipeline.

        Args:
            X: DataFrame with features
            y: Target values
        """
        print("[Preprocessing] Transforming features...")
        X_transformed = self.preprocessor.fit_transform(X)

        # Initialize models with pipeline
        print("[RF] Training Random Forest...")
        self.rf_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=200, max_depth=12,
                random_state=self.random_state, n_jobs=-1
            ))
        ])
        self.rf_model.fit(X, y)
        rf_score = self.rf_model.score(X, y)
        print(f"    Random Forest R² (train): {rf_score:.4f}")

        print("[XGB] Training XGBoost...")
        self.xgb_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', xgb.XGBRegressor(
                n_estimators=200, max_depth=6,
                learning_rate=0.1, random_state=self.random_state, n_jobs=-1
            ))
        ])
        self.xgb_model.fit(X, y)
        xgb_score = self.xgb_model.score(X, y)
        print(f"    XGBoost R² (train): {xgb_score:.4f}")

        print("[LGB] Training LightGBM...")
        self.lgb_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', lgb.LGBMRegressor(
                n_estimators=200, max_depth=7,
                learning_rate=0.1, random_state=self.random_state, n_jobs=-1
            ))
        ])
        self.lgb_model.fit(X, y)
        lgb_score = self.lgb_model.score(X, y)
        print(f"    LightGBM R² (train): {lgb_score:.4f}")

        print("[NN] Training Neural Network...")
        from sklearn.neural_network import MLPRegressor
        self.nn_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=500, random_state=self.random_state,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20
            ))
        ])
        self.nn_model.fit(X, y)
        nn_score = self.nn_model.score(X, y)
        print(f"    Neural Network R² (train): {nn_score:.4f}")

        self.models_trained = True
        print("\n[Ensemble] All models trained with preprocessing!")

    def predict(self, X):
        """Make ensemble predictions."""
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call fit() first.")

        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        nn_pred = self.nn_model.predict(X)

        ensemble_pred = (
            self.model_weights['rf'] * rf_pred +
            self.model_weights['xgb'] * xgb_pred +
            self.model_weights['lgb'] * lgb_pred +
            self.model_weights['nn'] * nn_pred
        )

        return ensemble_pred

    def predict_with_confidence(self, X):
        """Get predictions with confidence scores."""
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call fit() first.")

        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        nn_pred = self.nn_model.predict(X)

        ensemble_pred = (
            self.model_weights['rf'] * rf_pred +
            self.model_weights['xgb'] * xgb_pred +
            self.model_weights['lgb'] * lgb_pred +
            self.model_weights['nn'] * nn_pred
        )

        all_preds = np.array([rf_pred, xgb_pred, lgb_pred, nn_pred])
        confidence = np.std(all_preds, axis=0)

        return {
            'ensemble': ensemble_pred,
            'rf': rf_pred,
            'xgb': xgb_pred,
            'lgb': lgb_pred,
            'nn': nn_pred,
            'confidence': confidence,
            'agreement': 1.0 - (confidence / (np.max(all_preds, axis=0) - np.min(all_preds, axis=0) + 1e-8))
        }

    def evaluate(self, X, y):
        """Evaluate all models."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        nn_pred = self.nn_model.predict(X)
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
        """Save all models."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.preprocessor, path / 'preprocessor.joblib')
        joblib.dump(self.rf_model, path / 'rf_model.joblib')
        joblib.dump(self.xgb_model, path / 'xgb_model.joblib')
        joblib.dump(self.lgb_model, path / 'lgb_model.joblib')
        joblib.dump(self.nn_model, path / 'nn_model.joblib')
        joblib.dump(self.model_weights, path / 'weights.joblib')

        print(f"[OK] Models saved to {path}")

    def load(self, path):
        """Load all models."""
        path = Path(path)

        self.preprocessor = joblib.load(path / 'preprocessor.joblib')
        self.rf_model = joblib.load(path / 'rf_model.joblib')
        self.xgb_model = joblib.load(path / 'xgb_model.joblib')
        self.lgb_model = joblib.load(path / 'lgb_model.joblib')
        self.nn_model = joblib.load(path / 'nn_model.joblib')
        self.model_weights = joblib.load(path / 'weights.joblib')

        self.models_trained = True
        print(f"[OK] Models loaded from {path}")


# Example with your data
if __name__ == "__main__":
    print("Travel Recommendation Ensemble with Preprocessing Pipeline\n")

    # Example: your feature types
    numerical_features = ['temperature', 'rainfall', 'user_rating']
    categorical_features = ['climate_preference', 'budget', 'travel_month']

    ensemble = EnsembleRecommender(
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}\n")

    # Create sample data
    X = pd.DataFrame({
        'temperature': np.random.randn(100),
        'rainfall': np.random.randn(100),
        'user_rating': np.random.randn(100),
        'climate_preference': np.random.choice(['warm', 'moderate', 'cool'], 100),
        'budget': np.random.choice(['low', 'medium', 'high'], 100),
        'travel_month': np.random.randint(1, 13, 100)
    })
    y = np.random.randn(100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train
    ensemble.fit(X_train, y_train)

    # Evaluate
    print("\nModel Performance:")
    print(ensemble.evaluate(X_test, y_test))

    # Predict with confidence
    results = ensemble.predict_with_confidence(X_test.iloc[:5])
    print("\nSample Predictions with Confidence:")
    for key in ['ensemble', 'confidence', 'agreement']:
        print(f"  {key}: {results[key][:3]}")
