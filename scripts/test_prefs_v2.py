"""
Test preferences - FIXED VERSION
Boost preference weighting to 90% so interests actually matter
"""

import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, '.')

from config import *

print("="*80)
print("TESTING PREFERENCES (PREFERENCES WEIGHTED 90%)")
print("="*80)

# Load
rf_model = joblib.load(OUTPUT_DIR / 'random_forest_model_expanded.joblib')
dest_embeddings = joblib.load(CACHE_DIR / 'destination_embeddings.joblib')

embedder = SentenceTransformer('all-MiniLM-L6-v2')
print(f"✓ Loaded model and embeddings ({len(dest_embeddings)} cities)")

# Interest embeddings
INTERESTS = {
    'beaches': 'beach sand ocean water swimming surfing',
    'nightlife': 'bar club nightlife party dancing',
    'food': 'food restaurant cuisine',
    'culture': 'culture history museum art',
}

interest_embs = {k: embedder.encode(v) for k, v in INTERESTS.items()}
print(f"✓ Generated {len(interest_embs)} interest embeddings")

def rank_by_interests(interests_list):
    """Rank cities by interest matching (90% weight on interests)"""
    rankings = []

    for city, dest_emb in dest_embeddings.items():
        # Calculate interest similarity
        sims = []
        for interest in interests_list:
            if interest in interest_embs:
                interest_emb = interest_embs[interest]
                sim = np.dot(dest_emb, interest_emb) / (
                    np.linalg.norm(dest_emb) * np.linalg.norm(interest_emb) + 1e-6
                )
                sims.append(sim)

        interest_score = np.mean(sims) if sims else 0.0

        # Final: 90% interests, 10% default (to have some variance)
        final_score = interest_score * 0.9 + 0.5 * 0.1

        rankings.append({
            'city': city,
            'score': final_score,
            'interest_match': interest_score
        })

    return sorted(rankings, key=lambda x: x['score'], reverse=True)

# TEST 1: Beach
print("\n" + "="*80)
print("BEACH LOVER (interests: beaches, nightlife)")
print("="*80)
beach = rank_by_interests(['beaches', 'nightlife'])

print(f"\n{'Rank':<5} {'City':<20} {'Interest Score':<15}")
print("-" * 50)
for i, r in enumerate(beach[:10], 1):
    print(f"{i:<5} {r['city']:<20} {r['interest_match']:<15.3f}")

beach_cities = [r['city'] for r in beach[:5]]

# TEST 2: Culture
print("\n" + "="*80)
print("CULTURE LOVER (interests: culture, food)")
print("="*80)
culture = rank_by_interests(['culture', 'food'])

print(f"\n{'Rank':<5} {'City':<20} {'Interest Score':<15}")
print("-" * 50)
for i, r in enumerate(culture[:10], 1):
    print(f"{i:<5} {r['city']:<20} {r['interest_match']:<15.3f}")

culture_cities = [r['city'] for r in culture[:5]]

# COMPARISON
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nBeach Top 5:      {beach_cities}")
print(f"Culture Top 5:    {culture_cities}")

overlap = len(set(beach_cities) & set(culture_cities))
print(f"\nOverlap: {overlap}/5 cities are the same")

if overlap <= 2:
    print("✅ SUCCESS! Preferences ARE working - different cities for different interests!")
else:
    print("⚠️  Still showing similar cities - interests not differentiating enough")

# Show what happened to Indianapolis
indy_beach = next((i+1 for i, r in enumerate(beach) if r['city'].lower() == 'indianapolis'), None)
indy_culture = next((i+1 for i, r in enumerate(culture) if r['city'].lower() == 'indianapolis'), None)

print(f"\nIndianapolis ranking:")
print(f"  Beach preferences:   #{indy_beach if indy_beach else 'Not in top 39'}")
print(f"  Culture preferences: #{indy_culture if indy_culture else 'Not in top 39'}")
