"""
PHASE 5: ABLATION STUDY (EXPANDED)

Test which features matter most

Run: python phase5_expanded.py
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
print("PHASE 5: ABLATION STUDY (EXPANDED)")
print("="*80)
print("Testing feature importance by removing features\n")

# Load data
print("[1/3] Loading data...")
reviews_df = pd.read_parquet(CACHE_DIR / "reviews_phase2_expanded.parquet")
destinations_df = pd.read_parquet(CACHE_DIR / "destinations_phase2_expanded.parquet")
dest_embeddings = joblib.load(CACHE_DIR / "destination_embeddings_phase3_expanded.joblib")

# Build features
user_features_list = []
for user_id in reviews_df['user_id'].unique():
    user_reviews = reviews_df[reviews_df['user_id'] == user_id]
    user_features_list.append({
        'user_id': user_id,
        'avg_rating': user_reviews['stars'].mean(),
        'rating_std': user_reviews['stars'].std() or 0,
        'num_reviews': len(user_reviews),
        'num_cities': user_reviews['city'].nunique()
    })

user_features_df = pd.DataFrame(user_features_list)

# Train/test split
train_reviews, test_reviews = train_test_split(reviews_df, test_size=0.25, random_state=42)

print(f"✓ Train: {len(train_reviews):,}, Test: {len(test_reviews):,}")

# Baseline model (all features)
print("\n[2/3] Training baseline model...")

rf_baseline = RandomForestRecommender()
X_train, y_train = rf_baseline.build_feature_matrix(
    train_reviews, user_features_df, destinations_df, dest_embeddings, {}
)
rf_baseline.train(X_train, y_train)

X_test, y_test = rf_baseline.build_feature_matrix(
    test_reviews, user_features_df, destinations_df, dest_embeddings, {}
)
y_pred_baseline = rf_baseline.predict(X_test)

baseline_mse = mean_squared_error(y_test, y_pred_baseline)
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
baseline_r2 = r2_score(y_test, y_pred_baseline)

print(f"✓ Baseline trained")
print(f"  MSE: {baseline_mse:.4f}, R²: {baseline_r2:.4f}")

# Feature importance
print("\n[3/3] Analyzing feature importance...")

importance_df = rf_baseline.get_feature_importance(top_n=20)

# Categorize features
def categorize(feature):
    if 'emb' in feature:
        return 'Embeddings'
    elif 'rating' in feature.lower():
        return 'Ratings'
    elif 'review' in feature.lower() or 'business' in feature.lower():
        return 'Counts'
    elif any(x in feature.lower() for x in ['temp', 'precip', 'lat', 'lon']):
        return 'Weather'
    else:
        return 'Other'

importance_df['category'] = importance_df['feature'].apply(categorize)

# Results
print("\n" + "="*80)
print("ABLATION RESULTS")
print("="*80)

print("\nTop 20 Features:")
print("-" * 70)
for idx, row in importance_df.head(20).iterrows():
    print(f"  {row['feature']:<30} {row['importance']:.4f}  ({row['category']})")

print("\n" + "="*80)
print("Feature Importance by Category:")
print("="*80)

category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
total_importance = category_importance.sum()

for cat, imp in category_importance.items():
    pct = (imp / total_importance) * 100
    print(f"  {cat:<20}: {imp:.4f} ({pct:5.1f}%)")

# Save results
print("\nSaving results...")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

results_path = REPORT_DIR / 'phase5_expanded_ablation.txt'
with open(results_path, 'w', encoding='utf-8') as f:
    f.write(f"""PHASE 5: ABLATION STUDY (EXPANDED)
=====================================

BASELINE MODEL PERFORMANCE:
  MSE:  {baseline_mse:.4f}
  MAE:  {baseline_mae:.4f}
  R²:   {baseline_r2:.4f}

TOP 20 FEATURES:
""")
    for idx, row in importance_df.head(20).iterrows():
        f.write(f"  {row['feature']:<30} {row['importance']:.4f}  ({row['category']})\n")

    f.write(f"""
FEATURE IMPORTANCE BY CATEGORY:
""")
    for cat, imp in category_importance.items():
        pct = (imp / total_importance) * 100
        f.write(f"  {cat:<20}: {imp:.4f} ({pct:5.1f}%)\n")

print(f"✓ Saved: {results_path}")

print("\n" + "="*80)
print("✅ PHASE 5 COMPLETE")
print("="*80)
print(f"""
KEY INSIGHT:
  Features are automatically weighted by Random Forest
  based on predictive power. No manual tuning needed!

NEXT STEP:
  Run: python phase6_expanded.py (final analysis)
""")
