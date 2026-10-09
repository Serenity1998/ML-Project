"""
Expand to at least HALF of all cities in full Yelp dataset
"""

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

from config import *

print("="*80)
print("EXPANDING TO HALF OF ALL CITIES IN FULL YELP DATA")
print("="*80)

# Step 1: Load FULL business data
print("\n[1/5] Loading FULL Yelp businesses data...")
yelp_business_path = Path("raw_data/yelp_academic_dataset_business.json")

if not yelp_business_path.exists():
    print(f"❌ Business file not found at {yelp_business_path}")
    print("Looking for alternative paths...")
    alt_paths = [
        Path("data/raw/yelp_academic_dataset_business.json"),
        Path("yelp_academic_dataset_business.json"),
    ]
    for path in alt_paths:
        if path.exists():
            yelp_business_path = path
            print(f"✓ Found at: {path}")
            break
    else:
        print("❌ Could not find Yelp business data!")
        sys.exit(1)

businesses_list = []
with open(yelp_business_path, encoding='utf-8', errors='ignore') as f:
    for line in f:
        try:
            businesses_list.append(json.loads(line))
        except:
            continue

businesses_full = pd.DataFrame(businesses_list)
print(f"✓ Loaded {len(businesses_full):,} businesses")

# Get all unique cities
all_cities = businesses_full['city'].unique()
print(f"✓ Total unique cities: {len(all_cities)}")

# Take at least HALF of cities
cities_to_use = sorted(all_cities)[:len(all_cities)//2]
print(f"✓ Using {len(cities_to_use)} cities (at least half)")

# Step 2: Load expanded reviews and filter to these cities
print("\n[2/5] Loading expanded reviews and filtering to new cities...")
reviews_df = pd.read_parquet(CACHE_DIR / "reviews_expanded.parquet")
old_businesses = pd.read_parquet(CACHE_DIR / "businesses_major_cities.parquet")

# Merge reviews with business data to get cities
reviews_df = reviews_df.merge(
    old_businesses[['business_id', 'city']],
    on='business_id',
    how='left',
    suffixes=('', '_old')
)

# Also add from new businesses
reviews_df2 = pd.DataFrame()  # Will be filled with new city reviews

# Load full reviews
print("Loading FULL Yelp reviews...")
yelp_reviews_path = Path("raw_data/yelp_academic_dataset_review.json")

if not yelp_reviews_path.exists():
    alt_paths = [
        Path("data/raw/yelp_academic_dataset_review.json"),
        Path("yelp_academic_dataset_review.json"),
    ]
    for path in alt_paths:
        if path.exists():
            yelp_reviews_path = path
            break

reviews_new = []
with open(yelp_reviews_path, encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print(f"  Loaded {i:,} reviews...")
        try:
            reviews_new.append(json.loads(line))
        except:
            continue

reviews_new_df = pd.DataFrame(reviews_new)
print(f"✓ Loaded {len(reviews_new_df):,} total reviews")

# Merge with business data
reviews_new_df = reviews_new_df.merge(
    businesses_full[['business_id', 'city']],
    on='business_id',
    how='inner'
)

# Filter to target cities
reviews_new_df = reviews_new_df[reviews_new_df['city'].isin(cities_to_use)]
print(f"✓ Reviews in target cities: {len(reviews_new_df):,}")

# Step 3: Generate embeddings
print("\n[3/5] Generating embeddings for reviews...")

# Load Sentence Transformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Embedder loaded")

# Generate embeddings in batches
batch_size = 128
embeddings_list = []

print(f"Generating {len(reviews_new_df):,} embeddings...")
for i in tqdm(range(0, len(reviews_new_df), batch_size)):
    batch = reviews_new_df.iloc[i:i+batch_size]['text'].tolist()
    batch_embeddings = embedder.encode(batch, show_progress_bar=False)
    embeddings_list.append(batch_embeddings)

embeddings_new = np.vstack(embeddings_list)
print(f"✓ Generated {embeddings_new.shape[0]:,} embeddings")

# Step 4: Create city embeddings by averaging
print("\n[4/5] Creating city embeddings...")

dest_embeddings_all = {}

for city in tqdm(cities_to_use):
    city_reviews = reviews_new_df[reviews_new_df['city'] == city]

    if len(city_reviews) > 0:
        # Get indices of city reviews
        indices = city_reviews.index.tolist()

        # Get embeddings for these reviews
        city_embeddings = embeddings_new[[i for i in range(len(embeddings_new)) if i in indices]]

        if len(city_embeddings) > 0:
            # Average the embeddings
            city_embedding = np.mean(city_embeddings, axis=0)
            dest_embeddings_all[city] = city_embedding

print(f"✓ Created embeddings for {len(dest_embeddings_all)} cities")

# Step 5: Save
print("\n[5/5] Saving...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save embeddings
embeddings_path = OUTPUT_DIR / 'city_embeddings_expanded.joblib'
joblib.dump(dest_embeddings_all, embeddings_path)
print(f"✓ Saved embeddings: {embeddings_path}")

# Save cities list
cities_path = OUTPUT_DIR / 'expanded_cities.txt'
with open(cities_path, 'w') as f:
    for city in sorted(dest_embeddings_all.keys()):
        f.write(f"{city}\n")
print(f"✓ Saved cities list: {cities_path}")

print(f"\n{'='*80}")
print(f"✅ EXPANDED TO {len(dest_embeddings_all)} CITIES!")
print(f"{'='*80}")

print(f"\nSample cities:")
for city in sorted(list(dest_embeddings_all.keys()))[:20]:
    print(f"  - {city}")

print(f"\nNow update your app.py:")
print(f"  dest_embeddings = joblib.load(OUTPUT_DIR / 'city_embeddings_expanded.joblib')")

print(f"\nThen restart Flask and enjoy {len(dest_embeddings_all)} cities! 🌍")
