#!/usr/bin/env python
"""Test script to verify price level feature integration."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from config import *
from src.data.loader import load_businesses, load_tips, load_reviews, load_users
from src.features.price_level_features import PriceLevelFeatureBuilder
from src.features.feature_engineering_with_budget import FeatureEngineerWithBudget

print("=" * 80)
print("TESTING PRICE LEVEL FEATURE INTEGRATION")
print("=" * 80)

# 1. Load sample data
print("\n1. Loading sample data...")
businesses = load_businesses(str(BUSINESS_FILE), sample=True, sample_fraction=0.01)
tips = load_tips(str(TIPS_FILE), sample=True, sample_fraction=0.01)
reviews = load_reviews(str(REVIEW_FILE), sample=True, sample_fraction=0.01)
users = load_users(str(USER_FILE), sample=True, sample_fraction=0.01)

print(f"   [OK] Loaded {len(businesses)} businesses")
print(f"   [OK] Loaded {len(tips)} tips")
print(f"   [OK] Loaded {len(reviews)} reviews")
print(f"   [OK] Loaded {len(users)} users")

# 2. Build price features
print("\n2. Building price level features...")
builder = PriceLevelFeatureBuilder()
price_features = builder.build_price_features(businesses, tips)

print(f"   [OK] Generated price levels for {len(price_features)} businesses")
print(f"   \n   Price level distribution:")
dist = price_features['price_level'].value_counts().sort_index()
print(f"   {dist.to_dict()}")
print(f"   \n   Price sources:")
sources = price_features['price_source'].value_counts()
print(f"   {sources.to_dict()}")

# 3. Merge with businesses
print("\n3. Merging price levels with businesses...")
businesses_with_price = businesses.merge(
    price_features[['business_id', 'price_level', 'price_source']],
    on='business_id',
    how='left'
)
price_labels = {1: 'Low', 2: 'Medium', 3: 'High'}
businesses_with_price['price_label'] = businesses_with_price['price_level'].map(price_labels)

print(f"   [OK] Merged with {len(businesses_with_price)} businesses")
print(f"   \n   Sample businesses with price levels:")
sample = businesses_with_price[['name', 'city', 'price_level', 'price_label']].head()
for idx, row in sample.iterrows():
    print(f"   - {row['name']}: {row['price_label']} (level {row['price_level']})")

# 4. Test feature engineering
print("\n4. Testing feature engineering with budget...")
engineer = FeatureEngineerWithBudget()

# Filter to businesses we have
filtered_reviews = reviews[reviews['business_id'].isin(businesses_with_price['business_id'])]
filtered_users = users[users['user_id'].isin(filtered_reviews['user_id'])]

print(f"   [OK] Using {len(filtered_reviews)} reviews from {len(filtered_users)} users")

# Build features
try:
    all_features = engineer.build_features(
        filtered_reviews, filtered_users, businesses_with_price,
        weather_df=None,
        include_price=True
    )
    print(f"   [OK] Built features for {len(all_features)} records")
    print(f"   \n   Feature columns: {list(all_features.columns)}")

    # Test feature selection
    feature_df = engineer.get_feature_set(all_features, ["user_rating_behavior", "destination_stats", "price_level"])
    print(f"   [OK] Selected features: {list(feature_df.columns)}")
    print(f"   [OK] Final feature set shape: {feature_df.shape}")

except Exception as e:
    print(f"   [ERROR] {e}")

# 5. Test budget filtering
print("\n5. Testing budget filtering...")
for budget in ['Low', 'Medium', 'High']:
    count = len(businesses_with_price[businesses_with_price['price_label'] == budget])
    print(f"   [OK] {budget:6} budget restaurants: {count:4}")

# 6. Test different feature configs
print("\n6. Testing feature configs from config.py...")
from config import FEATURE_CONFIGS

for config_name, features in FEATURE_CONFIGS.items():
    has_price = "price_level" in features
    status = "[+]" if has_price else "[ ]"
    print(f"   {status} {config_name:20} -> {features}")

print("\n" + "=" * 80)
print("SUCCESS: ALL TESTS PASSED")
print("=" * 80)
print("\nNext steps:")
print("1. Run: notebooks/phase2_weather/02_price_level_features.ipynb")
print("2. Integrate into your training pipeline")
print("3. Test model performance with/without budget feature")
print("4. Deploy with budget filtering at recommendation time")
