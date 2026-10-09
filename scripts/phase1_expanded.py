"""
PHASE 1: DATA LOADING & PREPARATION (EXPANDED VERSION)
For teammates to run with full expanded dataset

Run: python phase1_expanded.py
"""

import json
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '.')

from config import *

print("="*80)
print("PHASE 1: DATA LOADING & PREPARATION (EXPANDED)")
print("="*80)

# Step 1: Load Yelp data
print("\n[1/3] Loading Yelp data...")

# Load businesses
print("  Loading businesses...")
business_path = Path("raw_data/yelp_academic_dataset_business.json")
if not business_path.exists():
    print(f"  ❌ {business_path} not found!")
    sys.exit(1)

businesses = []
with open(business_path, encoding='utf-8', errors='ignore') as f:
    for line in f:
        try:
            businesses.append(json.loads(line))
        except:
            continue

businesses_df = pd.DataFrame(businesses)
print(f"  ✓ Loaded {len(businesses_df):,} businesses")
print(f"  ✓ Cities: {len(businesses_df['city'].unique())}")

# Load reviews
print("  Loading reviews...")
reviews_path = Path("raw_data/yelp_academic_dataset_review.json")
if not reviews_path.exists():
    print(f"  ❌ {reviews_path} not found!")
    sys.exit(1)

reviews = []
for i, line in enumerate(open(reviews_path, encoding='utf-8', errors='ignore')):
    if i % 500000 == 0:
        print(f"    Loaded {i:,} reviews...")
    try:
        reviews.append(json.loads(line))
    except:
        continue

reviews_df = pd.DataFrame(reviews)
print(f"  ✓ Loaded {len(reviews_df):,} reviews")

# Step 2: Filter data
print("\n[2/3] Filtering data...")

# Merge to get cities
reviews_df = reviews_df.merge(
    businesses_df[['business_id', 'city']],
    on='business_id',
    how='inner'
)
print(f"  ✓ Reviews with city info: {len(reviews_df):,}")

# Get top 500 cities (half of all)
all_cities = businesses_df['city'].value_counts().index
top_cities = all_cities[:len(all_cities)//2]
print(f"  ✓ Using {len(top_cities)} cities")

# Filter reviews to top cities
reviews_df = reviews_df[reviews_df['city'].isin(top_cities)]
print(f"  ✓ Reviews in top cities: {len(reviews_df):,}")

# Filter to cross-city users (users who reviewed 2+ cities)
user_cities = reviews_df.groupby('user_id')['city'].nunique()
cross_city_users = user_cities[user_cities >= 2].index
reviews_df = reviews_df[reviews_df['user_id'].isin(cross_city_users)]

print(f"  ✓ Cross-city reviews: {len(reviews_df):,}")
print(f"  ✓ Unique users: {reviews_df['user_id'].nunique():,}")
print(f"  ✓ Unique cities: {reviews_df['city'].nunique():,}")

# Step 3: Save processed data
print("\n[3/3] Saving processed data...")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Save reviews
reviews_path_save = CACHE_DIR / "reviews_phase1_expanded.parquet"
reviews_df.to_parquet(reviews_path_save, index=False)
print(f"  ✓ Saved reviews: {reviews_path_save}")

# Save businesses (just the useful columns)
businesses_save = businesses_df[['business_id', 'city', 'name']].drop_duplicates()
businesses_save_path = CACHE_DIR / "businesses_phase1_expanded.parquet"
businesses_save.to_parquet(businesses_save_path, index=False)
print(f"  ✓ Saved businesses: {businesses_save_path}")

# Save cities list
cities_list = sorted(reviews_df['city'].unique())
cities_path_save = CACHE_DIR / "cities_phase1_expanded.txt"
with open(cities_path_save, 'w') as f:
    for city in cities_list:
        f.write(f"{city}\n")
print(f"  ✓ Saved cities list: {cities_path_save}")

print("\n" + "="*80)
print("✅ PHASE 1 COMPLETE")
print("="*80)
print(f"""
DATA SUMMARY:
  • Total reviews: {len(reviews_df):,}
  • Unique users: {reviews_df['user_id'].nunique():,}
  • Unique cities: {reviews_df['city'].nunique():,}
  • Review text available: ✓
  • Ratings (stars): ✓

FILES CREATED:
  • {reviews_path_save}
  • {businesses_save_path}
  • {cities_path_save}

NEXT STEP:
  Run: python phase2_expanded.py
""")
