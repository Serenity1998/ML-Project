import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Point to the project root
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

def deduplicate_cities():
    """Merge duplicate city names and standardize."""
    return {
        "St. Louis": "St. Louis",
        "Saint Louis": "St. Louis",
        "St. Petersburg": "St. Petersburg",
        "Saint Petersburg": "St. Petersburg",
    }

def load_city_table():
    """
    Load and cache city features: scalars (avg_rating, num_reviews, num_businesses),
    weather (12 months temp + precip, annual), and 384-dim embeddings.
    Returns DataFrame indexed by city name.
    """
    cache_file = CACHE_DIR / "cities_cache.parquet"

    # Load base cities metadata
    metadata_file = CACHE_DIR / "major_cities_metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)
    cities = pd.DataFrame([{"city": c} for c in metadata["major_cities"]])

    # Load weather features
    weather = pd.read_parquet(CACHE_DIR / "weather_features_for_models.parquet")
    cities = cities.merge(weather, on="city", how="left")

    # Compute city scalars from businesses and reviews
    businesses = pd.read_parquet(CACHE_DIR / "businesses_major_cities.parquet")
    reviews = pd.read_parquet(CACHE_DIR / "train_reviews.parquet")  # or any reviews subset

    # Merge reviews with business city info
    reviews_with_city = reviews.merge(
        businesses[["business_id", "city"]],
        on="business_id",
        how="left"
    )

    city_stats = []
    for city in cities["city"]:
        city_reviews = reviews_with_city[reviews_with_city["city"] == city]
        city_businesses = businesses[businesses["city"] == city]

        city_stats.append({
            "city": city,
            "dest_avg_rating": city_reviews["stars"].mean() if len(city_reviews) > 0 else 3.5,
            "dest_num_reviews": len(city_reviews),
            "dest_num_businesses": len(city_businesses),
        })

    stats_df = pd.DataFrame(city_stats)
    cities = cities.merge(stats_df, on="city", how="left")

    # Load destination embeddings (384-dim)
    dest_embeddings = joblib.load(CACHE_DIR / "destination_embeddings.joblib")
    embedding_dict = {}
    for city in cities["city"]:
        embedding_dict[city] = dest_embeddings.get(city, np.zeros(384, dtype=np.float32))

    cities["dest_embedding"] = cities["city"].map(lambda c: embedding_dict[c])

    # Set index
    cities.set_index("city", inplace=True)
    return cities

# Global cache
_CITY_TABLE = None

def get_city_table():
    """Lazy-load city table on first call."""
    global _CITY_TABLE
    if _CITY_TABLE is None:
        _CITY_TABLE = load_city_table()
    return _CITY_TABLE
