import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Point to the project root
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Built by scripts/build_city_table.py from the full Yelp dataset (~480 cities,
# real climate by coordinates). Used when present; else the original 39 cities.
CITY_TABLE_V2 = CACHE_DIR / "city_table_v2.parquet"
CITY_CATEGORIES_V2 = CACHE_DIR / "city_categories_v2.parquet"

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
    # Scorer and RF use dest_-prefixed names (dest_temp_month_7, dest_lat, ...).
    # Without this rename every lookup fell back to its default, so all cities
    # looked identical on weather.
    weather = weather.rename(columns={c: f"dest_{c}" for c in weather.columns if c != "city"})
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
    # Try expanded embeddings first (500+ cities), fall back to original (39 cities)
    expanded_embeddings_path = OUTPUT_DIR / "city_embeddings_expanded.joblib"
    if expanded_embeddings_path.exists():
        dest_embeddings = joblib.load(expanded_embeddings_path)
        print(f"[+] Loaded expanded embeddings ({len(dest_embeddings)} cities)")
    else:
        dest_embeddings = joblib.load(CACHE_DIR / "destination_embeddings.joblib")
        print(f"[+] Loaded original embeddings ({len(dest_embeddings)} cities)")

    embedding_dict = {}
    for city in cities["city"]:
        embedding_dict[city] = dest_embeddings.get(city, np.zeros(384, dtype=np.float32))

    cities["dest_embedding"] = cities["city"].map(lambda c: embedding_dict[c])

    # Set index
    cities.set_index("city", inplace=True)
    return cities

def load_city_table_v2():
    cities = pd.read_parquet(CITY_TABLE_V2)
    cities["dest_embedding"] = cities["dest_embedding"].map(lambda e: np.asarray(e, dtype=np.float32))
    print(f"[+] Loaded city table v2 ({len(cities)} cities, {cities['metro'].nunique()} metros)")
    return cities

def smoothed_category_share(counts, n_city, smoothing=20):
    """
    Share of each city's businesses in each category, shrunk toward the global
    share so 1 matching business out of 15 doesn't dominate.
    """
    global_share = counts.sum() / n_city.sum()
    return (counts + smoothing * global_share).div(n_city + smoothing, axis=0)

def load_city_profiles_v2(cities):
    counts = pd.read_parquet(CITY_CATEGORIES_V2).reindex(cities.index).fillna(0)
    return smoothed_category_share(counts, cities["dest_num_businesses"]), cities["avg_price"]

def load_city_profiles(city_names, smoothing=20):
    """
    Per-city business profiles from Yelp categories and price levels.

    Returns:
        category_share: DataFrame (city x category) with the smoothed share of
            businesses in each category. Small cities are shrunk toward the
            global share so 1 matching business out of 50 doesn't dominate.
        avg_price: Series (city) of mean Yelp price level (1-4), using only
            prices that came from Yelp rather than the filled-in default.
    """
    businesses = pd.read_parquet(CACHE_DIR / "businesses_with_price.parquet")
    businesses = businesses[businesses["city"].isin(city_names)]

    cats = (
        businesses[["city", "categories"]]
        .dropna()
        .assign(category=lambda d: d["categories"].str.split(","))
        .explode("category", ignore_index=True)
    )
    cats["category"] = cats["category"].str.strip()
    counts = pd.crosstab(cats["city"], cats["category"])
    n_city = businesses.groupby("city").size().reindex(counts.index)
    category_share = smoothed_category_share(counts, n_city, smoothing)
    category_share = category_share.reindex(city_names).fillna(category_share.mean())

    priced = businesses[businesses["price_source"] != "default"]
    avg_price = priced.groupby("city")["price_level"].mean()
    avg_price = avg_price.reindex(city_names).fillna(avg_price.mean())

    return category_share, avg_price

# Global cache
_CITY_TABLE = None
_CITY_PROFILES = None

def get_city_table():
    """Lazy-load city table on first call."""
    global _CITY_TABLE
    if _CITY_TABLE is None:
        _CITY_TABLE = load_city_table_v2() if CITY_TABLE_V2.exists() else load_city_table()
    return _CITY_TABLE

def get_city_profiles():
    """Lazy-load (category_share, avg_price) on first call."""
    global _CITY_PROFILES
    if _CITY_PROFILES is None:
        cities = get_city_table()
        if CITY_TABLE_V2.exists() and CITY_CATEGORIES_V2.exists():
            _CITY_PROFILES = load_city_profiles_v2(cities)
        else:
            _CITY_PROFILES = load_city_profiles(list(cities.index))
    return _CITY_PROFILES
