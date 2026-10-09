"""
PHASE 3: GENERATE EMBEDDINGS (EXPANDED)

Run: python phase3_expanded.py
Time: 20-30 minutes (GPU: 3-5 min, CPU: 30+ min)
"""

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
print("PHASE 3: GENERATE EMBEDDINGS (EXPANDED)")
print("="*80)

# Load data
print("\n[1/3] Loading reviews...")
reviews_df = pd.read_parquet(CACHE_DIR / "reviews_phase2_expanded.parquet")
destinations_df = pd.read_parquet(CACHE_DIR / "destinations_phase2_expanded.parquet")

print(f"✓ Reviews: {len(reviews_df):,}")
print(f"✓ Cities: {len(destinations_df)}")

# Load model
print("\n[2/3] Loading Sentence-BERT model...")
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Generate embeddings in batches
print(f"\nGenerating {len(reviews_df):,} review embeddings...")
print("(This may take 10-30 minutes depending on GPU/CPU)\n")

batch_size = 128
embeddings_list = []

for i in tqdm(range(0, len(reviews_df), batch_size), desc="Embedding"):
    batch = reviews_df.iloc[i:i+batch_size]['text'].tolist()
    batch_embeddings = embedder.encode(batch, show_progress_bar=False)
    embeddings_list.append(batch_embeddings)

embeddings = np.vstack(embeddings_list)
print(f"\n✓ Generated {embeddings.shape[0]:,} embeddings")
print(f"✓ Embedding dimension: {embeddings.shape[1]}")

# Create destination embeddings (average of review embeddings per city)
print("\n[3/3] Creating city embeddings...")

reviews_df['embedding_idx'] = range(len(reviews_df))
dest_embeddings = {}

for city in tqdm(destinations_df['city'], desc="Cities"):
    city_indices = reviews_df[reviews_df['city'] == city]['embedding_idx'].values

    if len(city_indices) > 0:
        city_embeddings = embeddings[city_indices]
        dest_embeddings[city] = np.mean(city_embeddings, axis=0)

print(f"✓ Created {len(dest_embeddings)} city embeddings")

# Save
print("\nSaving...")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Save review embeddings
np.save(CACHE_DIR / "review_embeddings_phase3_expanded.npy", embeddings)
print(f"✓ Saved review embeddings: {embeddings.shape}")

# Save destination embeddings
joblib.dump(dest_embeddings, CACHE_DIR / "destination_embeddings_phase3_expanded.joblib")
print(f"✓ Saved city embeddings: {len(dest_embeddings)} cities")

# Save review IDs for tracking
reviews_df['embedding_idx'].to_csv(CACHE_DIR / "review_embedding_ids_phase3.csv", index=False)

print("\n" + "="*80)
print("✅ PHASE 3 COMPLETE")
print("="*80)
print(f"""
EMBEDDINGS SUMMARY:
  • Review embeddings: {embeddings.shape}
  • City embeddings: {len(dest_embeddings)}
  • Dimension: 384D (Sentence-BERT)

FILES CREATED:
  • review_embeddings_phase3_expanded.npy
  • destination_embeddings_phase3_expanded.joblib

NEXT STEP:
  Run: python phase4_expanded.py (train model)
""")
