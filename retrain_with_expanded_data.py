"""
Retrain models with expanded dataset and compare results
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

print("="*70)
print("RETRAINING WITH EXPANDED DATASET")
print("="*70)

# Load expanded data
print("\n[1/5] Loading expanded dataset...")
reviews_df = pd.read_parquet(CACHE_DIR / "reviews_expanded.parquet")
embeddings = np.load(CACHE_DIR / "review_embeddings_expanded.npy")
businesses = pd.read_parquet(CACHE_DIR / "businesses_major_cities.parquet")

print(f"  ✓ Reviews: {len(reviews_df):,}")
print(f"  ✓ Embeddings: {embeddings.shape}")
print(f"  ✓ Users: {reviews_df['user_id'].nunique():,}")

# Prepare for modeling
print("\n[2/5] Preparing features...")

# Merge for city info
reviews_full = reviews_df.merge(
    businesses[['business_id', 'city']],
    on='business_id',
    how='left'
)

# Build feature dataframes (same as before)
user_features_list = []
for user_id in reviews_full['user_id'].unique():
    user_reviews = reviews_full[reviews_full['user_id'] == user_id]
    user_feat = {
        'user_id': user_id,
        'avg_rating': user_reviews['stars'].mean(),
        'rating_std': user_reviews['stars'].std() or 0,
        'num_reviews': len(user_reviews),
        'num_cities': user_reviews['city'].nunique()
    }
    user_features_list.append(user_feat)

user_features_df = pd.DataFrame(user_features_list)
print(f"  ✓ User features: {user_features_df.shape}")

# Destination features
dest_features_list = []
for city in businesses['city'].unique():
    city_reviews = reviews_full[reviews_full['city'] == city]
    city_businesses = businesses[businesses['city'] == city]

    dest_feat = {
        'city': city,
        'avg_rating': city_reviews['stars'].mean() if len(city_reviews) > 0 else 0,
        'num_reviews': len(city_reviews),
        'num_businesses': len(city_businesses),
    }
    dest_features_list.append(dest_feat)

dest_features_df = pd.DataFrame(dest_features_list)

# Merge with weather
weather_features = pd.read_parquet(CACHE_DIR / 'weather_features_for_models.parquet')
dest_features_df = dest_features_df.merge(
    weather_features, left_on='city', right_on='city', how='left'
)

print(f"  ✓ Destination features: {dest_features_df.shape}")

# Load embeddings
print("\n[3/5] Loading pre-computed embeddings...")

dest_embeddings = joblib.load(CACHE_DIR / 'destination_embeddings.joblib')
user_embeddings = joblib.load(CACHE_DIR / 'user_embeddings.joblib')

# Note: For expanded dataset, we'd need to regenerate user embeddings from expanded data
# For now, use existing embeddings for users that exist

print(f"  ✓ Destination embeddings: {len(dest_embeddings)}")
print(f"  ✓ User embeddings: {len(user_embeddings)}")

# Split data
print("\n[4/5] Train/test split...")

train_reviews, test_reviews = train_test_split(
    reviews_full,
    test_size=0.25,
    random_state=42
)

print(f"  ✓ Train set: {len(train_reviews):,}")
print(f"  ✓ Test set: {len(test_reviews):,}")

# Train Random Forest
print("\n[5/5] Training Random Forest...")

rf_recommender = RandomForestRecommender(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    random_state=RF_RANDOM_STATE
)

X_train, y_train = rf_recommender.build_feature_matrix(
    train_reviews,
    user_features_df,
    dest_features_df,
    dest_embeddings,
    user_embeddings
)

print(f"  Training set: {X_train.shape}")
rf_recommender.train(X_train, y_train)

X_test, y_test = rf_recommender.build_feature_matrix(
    test_reviews,
    user_features_df,
    dest_features_df,
    dest_embeddings,
    user_embeddings
)

print(f"  Test set: {X_test.shape}")

y_pred = rf_recommender.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*70)
print("RESULTS: EXPANDED DATASET")
print("="*70)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²:  {r2:.4f}")
print("="*70)

# Compare with original
print("\nCOMPARISON:")
print("-"*70)
print(f"{'Metric':<15} {'Original (298)':<20} {'Expanded ({:,})':<20} {'Improvement':<15}")
print("-"*70)

original_mse = 1.1461
original_mae = 0.8836
original_r2 = -0.2091

mse_improvement = ((original_mse - mse) / original_mse) * 100
mae_improvement = ((original_mae - mae) / original_mae) * 100
r2_improvement = r2 - original_r2

print(f"{'MSE':<15} {original_mse:<20.4f} {mse:<20.4f} {mse_improvement:>+13.1f}%")
print(f"{'MAE':<15} {original_mae:<20.4f} {mae:<20.4f} {mae_improvement:>+13.1f}%")
print(f"{'R²':<15} {original_r2:<20.4f} {r2:<20.4f} {r2_improvement:>+13.4f}")
print("-"*70)

print(f"\n✅ Dataset expansion:")
print(f"   {len(reviews_df):,} samples (vs 298 original)")
print(f"   {reviews_df['user_id'].nunique():,} users (vs 18 original)")
print(f"   R² improved by {r2_improvement:.4f}")

# Save results
REPORT_DIR.mkdir(parents=True, exist_ok=True)
results_path = REPORT_DIR / 'expanded_dataset_results.txt'
with open(results_path, 'w') as f:
    f.write("EXPANDED DATASET RESULTS\n")
    f.write("="*50 + "\n")
    f.write(f"Samples: {len(reviews_df):,}\n")
    f.write(f"Users: {reviews_df['user_id'].nunique():,}\n")
    f.write(f"MSE: {mse:.4f} (improvement: {mse_improvement:+.1f}%)\n")
    f.write(f"MAE: {mae:.4f} (improvement: {mae_improvement:+.1f}%)\n")
    f.write(f"R²: {r2:.4f} (improvement: {r2_improvement:+.4f})\n")

print(f"\n✓ Results saved to {results_path}")

# Save model
model_path = OUTPUT_DIR / 'random_forest_model_expanded.joblib'
joblib.dump(rf_recommender, model_path)
print(f"✓ Model saved to {model_path}")

print("\n" + "="*70)
print("✅ EXPANDED DATASET TRAINING COMPLETE")
print("="*70)
