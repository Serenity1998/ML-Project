"""
Build the candidate city table for the recommender from the full Yelp dataset.

For every Yelp city with at least MIN_BUSINESSES businesses (after cleaning up
names like "AMBLER", "Huntingdon Valley PA", "St. Louis" vs "Saint Louis"):
  - location (median business lat/lon), state, metro area
  - business count, review count, review-weighted avg stars, avg price level
  - per-category business counts (for activity matching)
  - monthly climate (mean temp in C, total precip in mm) from the Open-Meteo
    historical archive, averaged over CLIMATE_YEARS, looked up by coordinates
  - 384-dim destination embedding where one exists (mean of known otherwise)

Outputs (data/cache):
  city_table_v2.parquet       one row per city, indexed by "City, ST"
  city_categories_v2.parquet  city x category business counts

Run from the project root:  python scripts/build_city_table.py
"""
import json
import re
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_BUSINESS = PROJECT_ROOT / "data" / "raw" / "yelp_academic_dataset_business.json"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEATHER_CACHE = CACHE_DIR / "weather" / "open_meteo_monthly.json"

MIN_BUSINESSES = 10
METRO_RADIUS_DEG = 0.6          # ~60 km: suburbs closer than this share a metro
WEATHER_GRID_DEG = 0.25         # cities in the same ~25 km cell share climate
CLIMATE_YEARS = (2016, 2023)
WEATHER_BATCH = 5               # locations per Open-Meteo request (stays under rate limit)

STATE_SUFFIX = r"\s+(PA|NJ|FL|MO|IL|TN|IN|LA|AZ|NV|CA|ID|DE|AB)$"


def normalize_city(name):
    name = re.sub(r"\(.*?\)", "", str(name))
    name = re.sub(r"[.,]+$", "", name.strip())
    name = re.sub(STATE_SUFFIX, "", name, flags=re.I)
    name = re.sub(r"\s+", " ", name).strip().title()
    name = re.sub(r"^St\.? ", "Saint ", name)
    name = re.sub(r"^Mt\.? ", "Mount ", name)
    return name


def parse_price(attributes):
    if not isinstance(attributes, dict):
        return np.nan
    value = str(attributes.get("RestaurantsPriceRange2", "")).strip("'\" ")
    return float(value) if value in {"1", "2", "3", "4"} else np.nan


def load_businesses():
    b = pd.read_json(RAW_BUSINESS, lines=True)
    b["city_name"] = b["city"].map(normalize_city)
    # Same name in several states: keep the dominant state's businesses only
    main_state = b.groupby("city_name")["state"].agg(lambda s: s.mode()[0])
    b = b[b["state"] == b["city_name"].map(main_state)].copy()
    b["city_key"] = b["city_name"] + ", " + b["state"]
    counts = b["city_key"].value_counts()
    b = b[b["city_key"].isin(counts[counts >= MIN_BUSINESSES].index)].copy()
    b["price_level"] = b["attributes"].map(parse_price)
    return b


def city_scalars(b):
    b = b.assign(weighted_stars=b["stars"] * b["review_count"])
    g = b.groupby("city_key")
    cities = pd.DataFrame({
        "city_name": g["city_name"].first(),
        "state": g["state"].first(),
        "dest_lat": g["latitude"].median(),
        "dest_lon": g["longitude"].median(),
        "dest_num_businesses": g.size(),
        "total_review_count": g["review_count"].sum(),
        "yelp_avg_stars": g["weighted_stars"].sum() / g["review_count"].sum(),
        "avg_price": g["price_level"].mean(),
    })
    cities["avg_price"] = cities["avg_price"].fillna(cities["avg_price"].median())
    return cities


def assign_metros(cities):
    coords = cities[["dest_lat", "dest_lon"]].values
    labels = fcluster(linkage(coords, method="single"), METRO_RADIUS_DEG, criterion="distance")
    cities["metro_id"] = labels
    biggest = cities.sort_values("dest_num_businesses").groupby("metro_id")["city_name"].last()
    cities["metro"] = cities["metro_id"].map(biggest)
    return cities.drop(columns="metro_id")


def category_counts(b):
    cats = (
        b[["city_key", "categories"]].dropna()
        .assign(category=lambda d: d["categories"].str.split(","))
        .explode("category", ignore_index=True)
    )
    cats["category"] = cats["category"].str.strip()
    return pd.crosstab(cats["city_key"], cats["category"])


def fetch_monthly_climate(cells):
    """cells: list of (lat, lon) grid points -> {"lat,lon": {"temp": [12], "precip": [12]}}"""
    cache = json.loads(WEATHER_CACHE.read_text()) if WEATHER_CACHE.exists() else {}
    todo = [c for c in cells if f"{c[0]},{c[1]}" not in cache]
    start, end = CLIMATE_YEARS
    for i in range(0, len(todo), WEATHER_BATCH):
        batch = todo[i:i + WEATHER_BATCH]
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={','.join(str(c[0]) for c in batch)}"
            f"&longitude={','.join(str(c[1]) for c in batch)}"
            f"&start_date={start}-01-01&end_date={end}-12-31"
            "&daily=temperature_2m_mean,precipitation_sum&timezone=auto"
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    data = json.loads(resp.read())
                break
            except Exception as e:  # rate limit or transient network error
                wait = 30 * (attempt + 1)
                print(f"  weather request failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError("Open-Meteo requests keep failing")
        data = data if isinstance(data, list) else [data]
        for cell, loc in zip(batch, data):
            daily = pd.DataFrame(loc["daily"])
            daily["time"] = pd.to_datetime(daily["time"])
            daily["year"], daily["month"] = daily["time"].dt.year, daily["time"].dt.month
            temp = daily.groupby("month")["temperature_2m_mean"].mean()
            precip = daily.groupby(["year", "month"])["precipitation_sum"].sum().groupby("month").mean()
            cache[f"{cell[0]},{cell[1]}"] = {"temp": temp.round(2).tolist(), "precip": precip.round(1).tolist()}
        WEATHER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        WEATHER_CACHE.write_text(json.dumps(cache))
        print(f"  weather: {min(i + WEATHER_BATCH, len(todo))}/{len(todo)} new locations")
        time.sleep(15)
    return cache


def add_climate(cities):
    grid = lambda v: round(round(v / WEATHER_GRID_DEG) * WEATHER_GRID_DEG, 2)
    cities["cell"] = [(grid(lat), grid(lon)) for lat, lon in zip(cities["dest_lat"], cities["dest_lon"])]
    climate = fetch_monthly_climate(sorted(set(cities["cell"])))
    for m in range(1, 13):
        cities[f"dest_temp_month_{m}"] = [climate[f"{c[0]},{c[1]}"]["temp"][m - 1] for c in cities["cell"]]
        cities[f"dest_precip_month_{m}"] = [climate[f"{c[0]},{c[1]}"]["precip"][m - 1] for c in cities["cell"]]
    temps = cities[[f"dest_temp_month_{m}" for m in range(1, 13)]]
    cities["dest_annual_avg_temp"] = temps.mean(axis=1)
    cities["dest_temp_range"] = temps.max(axis=1) - temps.min(axis=1)
    cities["dest_annual_precipitation"] = cities[[f"dest_precip_month_{m}" for m in range(1, 13)]].sum(axis=1)
    return cities.drop(columns="cell")


def add_training_stats(cities):
    """Ratings/review counts the RF was trained on (from the training split)."""
    businesses = pd.read_parquet(CACHE_DIR / "businesses_major_cities.parquet")[["business_id", "city"]]
    reviews = pd.read_parquet(CACHE_DIR / "train_reviews.parquet").merge(businesses, on="business_id")
    reviews["city_name"] = reviews["city"].map(normalize_city)
    stats = reviews.groupby("city_name")["stars"].agg(["mean", "size"])
    cities["dest_avg_rating"] = cities["city_name"].map(stats["mean"]).fillna(cities["yelp_avg_stars"])
    cities["dest_num_reviews"] = cities["city_name"].map(stats["size"]).fillna(0).astype(int)
    return cities


def add_embeddings(cities):
    known = {}
    for path in [CACHE_DIR / "destination_embeddings.joblib", OUTPUT_DIR / "city_embeddings_expanded.joblib"]:
        if path.exists():
            for name, emb in joblib.load(path).items():
                if np.any(emb):
                    known.setdefault(normalize_city(name), np.asarray(emb, dtype=np.float32))
    mean_emb = np.mean(list(known.values()), axis=0).astype(np.float32)
    cities["has_embedding"] = cities["city_name"].isin(known.keys())
    cities["dest_embedding"] = [known.get(n, mean_emb).tolist() for n in cities["city_name"]]
    return cities


def main():
    print("[1/5] Loading Yelp businesses...")
    b = load_businesses()
    cities = city_scalars(b)
    print(f"      {len(cities)} cities with >= {MIN_BUSINESSES} businesses")

    print("[2/5] Grouping into metro areas...")
    cities = assign_metros(cities)
    print(f"      {cities['metro'].nunique()} metros: {sorted(cities['metro'].value_counts().head(15).index)}")

    print("[3/5] Fetching monthly climate (Open-Meteo, cached)...")
    cities = add_climate(cities)

    print("[4/5] Adding RF training stats and embeddings...")
    cities = add_training_stats(cities)
    cities = add_embeddings(cities)
    print(f"      {cities['has_embedding'].sum()} cities have their own embedding")

    print("[5/5] Saving...")
    cities.index.name = "city"
    cities.to_parquet(CACHE_DIR / "city_table_v2.parquet")
    category_counts(b).reindex(cities.index).fillna(0).astype(int).to_parquet(CACHE_DIR / "city_categories_v2.parquet")
    print(f"      wrote city_table_v2.parquet and city_categories_v2.parquet ({len(cities)} cities)")


if __name__ == "__main__":
    main()
