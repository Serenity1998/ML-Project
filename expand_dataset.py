"""
Expand ML Project Dataset: Generate embeddings for full Yelp data
Time: ~4-6 hours with GPU, ~8-12 hours with CPU
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import joblib
from tqdm import tqdm
import time

# Config
YELP_JSON_PATH = Path("raw_data/yelp_academic_dataset_review.json")  # Adjust path
CACHE_DIR = Path("data/cache")
OUTPUT_DIR = Path("outputs")

print("="*70)
print("EXPANDING DATASET: Full Yelp Embedding Generation")
print("="*70)

# Step 1: Load and process Yelp JSON
print("\n[1/5] Loading Yelp JSON data...")
start = time.time()

reviews_list = []
businesses_list = []

# Load reviews from JSON (streaming to avoid memory issues)
try:
    with open(YELP_JSON_PATH) as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print(f"  Loaded {i:,} reviews...")
            reviews_list.append(json.loads(line))
except FileNotFoundError:
    # Try alternative path
    alt_path = Path("raw_data") / "yelp_academic_dataset_review.json"
    print(f"File not found at {YELP_JSON_PATH}")
    print(f"Looking for: {alt_path}")
    raise

reviews_df = pd.DataFrame(reviews_list)
print(f"  ✓ Total reviews loaded: {len(reviews_df):,}")
print(f"  Time: {time.time()-start:.1f}s")

# Step 2: Filter to major cities and cross-city users
print("\n[2/5] Filtering to major cities...")
start = time.time()

# Load businesses to get city info
businesses_path = Path("raw_data/yelp_academic_dataset_business.json")
try:
    with open(businesses_path) as f:
        businesses_list = [json.loads(line) for line in f]
    businesses_df = pd.DataFrame(businesses_list)
    print(f"  ✓ Loaded {len(businesses_df):,} businesses")
except:
    print("  ⚠️ Using cached businesses from previous run")
    businesses_df = pd.read_parquet(CACHE_DIR / "businesses_major_cities.parquet")

# Filter to major cities (top 39 by review count)
city_counts = businesses_df['city'].value_counts().head(39)
major_cities = set(city_counts.index)
print(f"  ✓ Major cities: {len(major_cities)}")

# Filter reviews to businesses in major cities
reviews_df = reviews_df.merge(
    businesses_df[['business_id', 'city']],
    on='business_id',
    how='inner'
)
print(f"  ✓ Reviews in major cities: {len(reviews_df):,}")

# Filter to cross-city users (users who reviewed in 2+ cities)
user_cities = reviews_df.groupby('user_id')['city'].nunique()
cross_city_users = user_cities[user_cities >= 2].index
reviews_df = reviews_df[reviews_df['user_id'].isin(cross_city_users)]
print(f"  ✓ Cross-city reviews: {len(reviews_df):,}")
print(f"  ✓ Unique users: {reviews_df['user_id'].nunique():,}")
print(f"  Time: {time.time()-start:.1f}s")

# Step 3: Generate embeddings
print("\n[3/5] Generating Sentence-BERT embeddings...")
print("  Loading model (first time takes ~2min)...")
start = time.time()

model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"  ✓ Model loaded in {time.time()-start:.1f}s")

# Generate embeddings in batches
batch_size = 128  # Adjust based on GPU memory
print(f"  Batch size: {batch_size}")
print(f"  Total reviews to embed: {len(reviews_df):,}")

embeddings_list = []
start = time.time()

# Process in batches
for i in tqdm(range(0, len(reviews_df), batch_size), desc="Embedding"):
    batch = reviews_df.iloc[i:i+batch_size]['text'].tolist()
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    embeddings_list.append(batch_embeddings)

embeddings = np.vstack(embeddings_list)
print(f"  ✓ Generated {embeddings.shape[0]:,} embeddings")
print(f"  ✓ Embedding dimension: {embeddings.shape[1]}")
print(f"  Time: {time.time()-start:.1f}s")

# Step 4: Save expanded dataset
print("\n[4/5] Saving expanded dataset...")
start = time.time()

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Save reviews with embeddings
reviews_df['embedding_id'] = range(len(reviews_df))
reviews_df.to_parquet(CACHE_DIR / "reviews_expanded.parquet", index=False)
print(f"  ✓ Saved reviews to reviews_expanded.parquet")

# Save embeddings as numpy
np.save(CACHE_DIR / "review_embeddings_expanded.npy", embeddings)
print(f"  ✓ Saved embeddings to review_embeddings_expanded.npy")

# Summary
print(f"\n  Dataset expanded from 298 → {len(reviews_df):,} reviews")
print(f"  Time: {time.time()-start:.1f}s")

# Step 5: Print summary
print("\n[5/5] Summary")
print("="*70)
print(f"Original dataset:  221 train + 77 test = 298 total")
print(f"Expanded dataset:  {len(reviews_df):,} total reviews")
print(f"Train/test split:  {int(len(reviews_df)*0.75):,} train + {int(len(reviews_df)*0.25):,} test")
print(f"Unique users:      {reviews_df['user_id'].nunique():,}")
print(f"Unique cities:     {reviews_df['city'].nunique():,}")
print(f"Embedding size:    {embeddings.shape}")
print("="*70)

print("\n✅ NEXT STEPS:")
print("1. Run: python retrain_with_expanded_data.py")
print("2. This will retrain RF and NN with expanded dataset")
print("3. Compare old vs new performance")
