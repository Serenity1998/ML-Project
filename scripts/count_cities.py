"""
Count and display all cities
"""

import joblib
from pathlib import Path
import sys
sys.path.insert(0, '.')

from config import *

# Load embeddings
dest_embeddings = joblib.load(OUTPUT_DIR / 'city_embeddings_all.joblib')

print("="*80)
print("ALL CITIES IN YOUR SYSTEM")
print("="*80)

cities = sorted(list(dest_embeddings.keys()))

print(f"\n✅ Total cities: {len(cities)}\n")

for i, city in enumerate(cities, 1):
    print(f"{i:3d}. {city}")

print(f"\n{'='*80}")
print(f"✅ YOUR SYSTEM HAS {len(cities)} CITIES!")
print(f"{'='*80}")
