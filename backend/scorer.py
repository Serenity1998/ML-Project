import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Add project root to sys.path for pickled model imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Global state
_MODEL = None
_EMBEDDING_MODEL = None
_CITY_TABLE = None

def load_model():
    """Load Random Forest recommender and embeddings."""
    global _MODEL, _EMBEDDING_MODEL, _CITY_TABLE

    if _MODEL is None:
        _MODEL = joblib.load(OUTPUT_DIR / "random_forest_model.joblib")
        _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

        # Load city table
        try:
            from backend.city_table import get_city_table
        except ImportError:
            from city_table import get_city_table
        _CITY_TABLE = get_city_table()

    return _MODEL, _EMBEDDING_MODEL, _CITY_TABLE

def compute_user_embedding(interests_text):
    """Convert user interests (free text) to 384-dim embedding."""
    _, emb_model, _ = load_model()
    if not interests_text or interests_text.strip() == "":
        return np.zeros(384, dtype=np.float32)
    return emb_model.encode(interests_text, convert_to_numpy=True).astype(np.float32)

def score_cities(user_prefs):
    """
    Score all candidate cities for the given user preferences.

    Args:
        user_prefs: dict with keys:
            - travel_month: int 1-12
            - climate_preference: str (e.g., "warm", "cool", "moderate")
            - activities: list of str (e.g., ["hiking", "museums"])
            - budget: str (e.g., "low", "medium", "high")
            - free_text_interests: str (free text about interests)

    Returns:
        DataFrame with columns [city, rf_score, climate_match, activity_sim, popularity, combined_score]
        sorted by combined_score descending
    """
    rf_model, emb_model, city_table = load_model()

    travel_month = user_prefs.get("travel_month", 6)  # default June
    activities_text = " ".join(user_prefs.get("activities", []))
    interests_text = user_prefs.get("free_text_interests", "")
    combined_text = f"{activities_text} {interests_text}"

    # Build pseudo-user features
    user_avg_rating = 3.5
    user_rating_std = 1.0
    user_num_reviews = 1
    user_num_cities = 1
    user_embedding = compute_user_embedding(combined_text)

    # Build feature matrix for all cities
    results = []
    for city_name, city_row in city_table.iterrows():
        # Extract destination features
        dest_avg_rating = city_row.get("dest_avg_rating", 3.5)
        dest_num_reviews = city_row.get("dest_num_reviews", 0)
        dest_num_businesses = city_row.get("dest_num_businesses", 0)
        dest_embedding = city_row.get("dest_embedding", np.zeros(384))

        # Weather for the travel month
        temp_col = f"dest_temp_month_{travel_month}"
        precip_col = f"dest_precip_month_{travel_month}"
        dest_temp = city_row.get(temp_col, 70)
        dest_precip = city_row.get(precip_col, 0)

        # Climate match score (rough heuristic)
        climate_pref = user_prefs.get("climate_preference", "moderate").lower()
        if climate_pref == "warm" and dest_temp > 75:
            climate_match = 0.8
        elif climate_pref == "cool" and dest_temp < 60:
            climate_match = 0.8
        elif climate_pref == "moderate" and 60 <= dest_temp <= 75:
            climate_match = 0.8
        else:
            climate_match = 0.5

        # Activity similarity (embedding cosine sim with destination embedding)
        if combined_text.strip():
            activity_embedding = emb_model.encode(combined_text, convert_to_numpy=True).astype(np.float32)
            activity_sim = np.dot(activity_embedding, dest_embedding) / (
                np.linalg.norm(activity_embedding) * np.linalg.norm(dest_embedding) + 1e-8
            )
            activity_sim = float(np.clip(activity_sim, 0, 1))
        else:
            activity_sim = 0.5

        # Popularity (log of reviews)
        popularity = np.log1p(dest_num_reviews) / np.log1p(1000)  # normalize to ~[0, 1]

        # Build full feature vector for RF
        feature_dict = {
            "user_avg_rating": user_avg_rating,
            "user_num_cities": user_num_cities,
            "user_num_reviews": user_num_reviews,
            "user_rating_std": user_rating_std,
            "dest_avg_rating": dest_avg_rating,
            "dest_lat": city_row.get("dest_lat", 0),
            "dest_lon": city_row.get("dest_lon", 0),
            "dest_num_businesses": dest_num_businesses,
            "dest_num_reviews": dest_num_reviews,
            "dest_annual_avg_temp": city_row.get("dest_annual_avg_temp", 70),
            "dest_annual_precipitation": city_row.get("dest_annual_precipitation", 0),
            "dest_temp_range": city_row.get("dest_temp_range", 30),
        }

        # Add weather for all 12 months
        for m in range(1, 13):
            feature_dict[f"dest_precip_month_{m}"] = city_row.get(f"dest_precip_month_{m}", 0)
            feature_dict[f"dest_temp_month_{m}"] = city_row.get(f"dest_temp_month_{m}", 70)

        # Add embeddings
        for i in range(384):
            feature_dict[f"dest_emb_{i}"] = dest_embedding[i] if i < len(dest_embedding) else 0.0
            feature_dict[f"user_emb_{i}"] = user_embedding[i] if i < len(user_embedding) else 0.0

        # Reindex on RF feature names (alphabetical order)
        X_df = pd.DataFrame([feature_dict])
        X_df = X_df.reindex(columns=sorted(feature_dict.keys()), fill_value=0)

        # Predict RF score (expected rating 1-5)
        rf_score = float(rf_model.predict(X_df.values)[0])
        rf_score = np.clip(rf_score, 1, 5)  # Clamp to valid rating range

        results.append({
            "city": city_name,
            "rf_score": rf_score,
            "climate_match": climate_match,
            "activity_sim": activity_sim,
            "popularity": float(popularity),
        })

    results_df = pd.DataFrame(results)

    # Normalize scores for combination
    results_df["rf_score_norm"] = (results_df["rf_score"] - results_df["rf_score"].min()) / (
        results_df["rf_score"].max() - results_df["rf_score"].min() + 1e-8
    )

    # Combined score (weighted average)
    results_df["combined_score"] = (
        0.5 * results_df["rf_score_norm"] +
        0.2 * results_df["climate_match"] +
        0.2 * results_df["activity_sim"] +
        0.1 * results_df["popularity"]
    )

    results_df = results_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return results_df
