"""
Simple: Extract all cities from embeddings and save
"""

import joblib
from pathlib import Path
import sys
sys.path.insert(0, '.')

from config import *

print("="*80)
print("EXTRACTING ALL CITIES FROM EMBEDDINGS")
print("="*80)

# Load current embeddings (which has all city embeddings)
print("\nLoading embeddings...")
dest_embeddings = joblib.load(CACHE_DIR / 'destination_embeddings.joblib')

print(f"✓ Found {len(dest_embeddings)} cities in embeddings")

# Save to outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
embeddings_path = OUTPUT_DIR / 'city_embeddings_all.joblib'
joblib.dump(dest_embeddings, embeddings_path)

print(f"✓ Saved to: {embeddings_path}")

# Save city list
cities_list = sorted(list(dest_embeddings.keys()))
cities_path = OUTPUT_DIR / 'all_cities.txt'
with open(cities_path, 'w') as f:
    for city in cities_list:
        f.write(f"{city}\n")

print(f"✓ Cities list: {cities_path}")

print(f"\n{'='*80}")
print(f"✅ ALL {len(dest_embeddings)} CITIES EXTRACTED!")
print(f"{'='*80}")

print(f"\nCities included:")
for city in cities_list[:10]:
    print(f"  - {city}")
print(f"  ... and {len(cities_list) - 10} more")

print(f"\nNow update your app.py to load:")
print(f"  dest_embeddings = joblib.load(OUTPUT_DIR / 'city_embeddings_all.joblib')")
print(f"\nThen restart the Flask server!")
