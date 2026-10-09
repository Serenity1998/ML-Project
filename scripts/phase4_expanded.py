"""
PHASE 4: TRAIN RANDOM FOREST MODEL (EXPANDED)

Run: python phase4_expanded.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.path.insert(0, '.')

from config import *
from src.models.random_forest import RandomForestRecommender

print("="*80)
print("PHASE 4: TRAIN RANDOM FOREST MODEL (EXPANDED)")
print("="*80)

# Load data
print("\n[1/4] Loading data...")
reviews_df = pd.read_parquet(CACHE_DIR / "reviews_phase2_expanded.parquet")
destinations_df = pd.read_parquet(CACHE_DIR / "destinations_phase2_expanded.parquet")

print(f"✓ Reviews: {len(reviews_df):,}")
print(f"✓ Destinations: {len(destinations_df)}")

# Load embeddings
print("\n[2/4] Loading embeddings...")
dest_embeddings = joblib.load(CACHE_DIR / "destination_embeddings_phase3_expanded.joblib")
embeddings = np.load(CACHE_DIR / "review_embeddings_phase3_expanded.npy")

print(f"✓ Destination embeddings: {len(dest_embeddings)}")
print(f"✓ Review embeddings: {embeddings.shape}")

# Build user features
print("\n[3/4] Building features...")

user_features_list = []
for user_id in reviews_df['user_id'].unique():
    user_reviews = reviews_df[reviews_df['user_id'] == user_id]
    user_feat = {
        'user_id': user_id,
        'avg_rating': user_reviews['stars'].mean(),
        'rating_std': user_reviews['stars'].std() or 0,
        'num_reviews': len(user_reviews),
        'num_cities': user_reviews['city'].nunique()
    }
    user_features_list.append(user_feat)

user_features_df = pd.DataFrame(user_features_list)
print(f"✓ User features: {user_features_df.shape}")

# Train/test split
train_reviews, test_reviews = train_test_split(
    reviews_df,
    test_size=0.25,
    random_state=42
)

print(f"✓ Train set: {len(train_reviews):,}")
print(f"✓ Test set: {len(test_reviews):,}")

# Train Random Forest
print("\nTraining Random Forest...")

rf_model = RandomForestRecommender(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    random_state=RF_RANDOM_STATE
)

# Build features
X_train, y_train = rf_model.build_feature_matrix(
    train_reviews,
    user_features_df,
    destinations_df,
    dest_embeddings,
    {}  # Empty user_embeddings (not needed for this phase)
)

print(f"✓ Training features: {X_train.shape}")

# Train
rf_model.train(X_train, y_train)
print("✓ Model trained")

# Evaluate
X_test, y_test = rf_model.build_feature_matrix(
    test_reviews,
    user_features_df,
    destinations_df,
    dest_embeddings,
    {}
)

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✓ Evaluation complete")

# Results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"""
MSE:  {mse:.4f}
MAE:  {mae:.4f}
R²:   {r2:.4f}

Dataset Size:  {len(reviews_df):,} reviews
Training:      {len(train_reviews):,} samples
Testing:       {len(test_reviews):,} samples
Features:      {X_train.shape[1]}
""")

# Feature importance
print("Top 10 Important Features:")
print("-" * 60)
importance_df = rf_model.get_feature_importance(top_n=10)
for idx, row in importance_df.iterrows():
    print(f"  {row['feature']:<30} {row['importance']:.4f}")

# Save model
print("\n[4/4] Saving model...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_path = OUTPUT_DIR / 'random_forest_model_phase4_expanded.joblib'
joblib.dump(rf_model, model_path)
print(f"✓ Model saved: {model_path}")

# Save results
REPORT_DIR.mkdir(parents=True, exist_ok=True)
results_path = REPORT_DIR / 'phase4_expanded_results.txt'
with open(results_path, 'w', encoding='utf-8') as f:
    f.write(f"""PHASE 4: RANDOM FOREST MODEL (EXPANDED)
===============================================

Dataset:       {len(reviews_df):,} reviews, {len(destinations_df)} cities
Train/Test:    {len(train_reviews):,} / {len(test_reviews):,}
Features:      {X_train.shape[1]}

RESULTS:
  MSE: {mse:.4f}
  MAE: {mae:.4f}
  R²:  {r2:.4f}

Model saved: {model_path}
""")

print(f"✓ Results saved: {results_path}")

print("\n" + "="*80)
print("✅ PHASE 4 COMPLETE")
print("="*80)
print(f"""
NEXT STEP:
  Run: python phase5_expanded.py (ablation study)
""")
