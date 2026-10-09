"""
PHASE 2: DESTINATION FILTERING & ANALYSIS (EXPANDED)

Run: python phase2_expanded.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

from config import *

print("="*80)
print("PHASE 2: DESTINATION FILTERING & ANALYSIS (EXPANDED)")
print("="*80)

# Load data from Phase 1
print("\n[1/4] Loading data from Phase 1...")
reviews_df = pd.read_parquet(CACHE_DIR / "reviews_phase1_expanded.parquet")
businesses_df = pd.read_parquet(CACHE_DIR / "businesses_phase1_expanded.parquet")

print(f"✓ Reviews: {len(reviews_df):,}")
print(f"✓ Cities: {reviews_df['city'].nunique()}")

# Step 1: Analyze destinations
print("\n[2/4] Analyzing destinations...")

dest_analysis = reviews_df.groupby('city').agg({
    'stars': ['mean', 'std', 'count'],
    'user_id': 'nunique',
    'business_id': 'nunique'
}).round(2)

dest_analysis.columns = ['avg_rating', 'rating_std', 'num_reviews', 'num_users', 'num_businesses']
dest_analysis = dest_analysis.reset_index()

print(f"\nTop 10 cities by reviews:")
print(dest_analysis.nlargest(10, 'num_reviews')[['city', 'num_reviews', 'avg_rating', 'num_users']])

# Step 2: Filter to cities with enough data
print("\n[3/4] Filtering to cities with sufficient data...")

# Keep cities with 50+ reviews
min_reviews = 50
dest_filtered = dest_analysis[dest_analysis['num_reviews'] >= min_reviews]
print(f"✓ Cities with {min_reviews}+ reviews: {len(dest_filtered)}")

# Filter reviews to these cities
valid_cities = set(dest_filtered['city'])
reviews_filtered = reviews_df[reviews_df['city'].isin(valid_cities)]

print(f"✓ Reviews in valid cities: {len(reviews_filtered):,}")
print(f"✓ Users: {reviews_filtered['user_id'].nunique():,}")
print(f"✓ Cities: {reviews_filtered['city'].nunique():,}")

# Step 3: Save filtered data
print("\n[4/4] Saving filtered data...")

reviews_filtered.to_parquet(CACHE_DIR / "reviews_phase2_expanded.parquet", index=False)
dest_filtered.to_parquet(CACHE_DIR / "destinations_phase2_expanded.parquet", index=False)

print(f"✓ Saved filtered reviews: {len(reviews_filtered):,}")
print(f"✓ Saved destinations: {len(dest_filtered)}")

print("\n" + "="*80)
print("✅ PHASE 2 COMPLETE")
print("="*80)
print(f"""
DESTINATION SUMMARY:
  • Valid cities: {len(dest_filtered)}
  • Total reviews: {len(reviews_filtered):,}
  • Total users: {reviews_filtered['user_id'].nunique():,}
  • Avg reviews per city: {len(reviews_filtered) / len(dest_filtered):.0f}

TOP 5 CITIES:
""")

for idx, row in dest_filtered.nlargest(5, 'num_reviews').iterrows():
    print(f"  • {row['city']:<20} {int(row['num_reviews']):>6} reviews  Rating: {row['avg_rating']:.2f}")

print(f"""
FILES CREATED:
  • reviews_phase2_expanded.parquet
  • destinations_phase2_expanded.parquet

NEXT STEP:
  Run: python phase3_expanded.py (generates embeddings)
""")
