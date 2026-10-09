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

# Ideal travel-month temperature (deg C, the unit of the weather data) for each
# climate preference, and how many degrees away halves-ish the match.
CLIMATE_TARGETS_C = {"warm": 28.0, "moderate": 21.0, "cool": 12.0}
CLIMATE_TOLERANCE_C = 6.0

# Form activities -> Yelp business categories that signal them.
ACTIVITY_CATEGORIES = {
    "hiking": ["Hiking", "Parks", "Climbing", "Rock Climbing", "Campgrounds", "Dog Parks", "Rafting/Kayaking"],
    "beaches": ["Beaches", "Boating", "Boat Charters", "Boat Tours", "Swimming Pools", "Swimwear", "Seafood", "Tiki Bars"],
    "museums": ["Museums", "Art Galleries", "Aquariums", "Zoos", "Landmarks & Historical Buildings"],
    "food/dining": ["Restaurants", "Food", "Specialty Food", "Food Trucks", "Food Tours", "Seafood", "Barbeque", "Soul Food", "Cajun/Creole"],
    "shopping": ["Shopping", "Shopping Centers", "Outlet Stores", "Department Stores", "Fashion", "Gift Shops", "Souvenir Shops"],
    "nightlife": ["Nightlife", "Bars", "Cocktail Bars", "Dance Clubs", "Music Venues", "Lounges", "Comedy Clubs", "Dive Bars"],
    "parks": ["Parks", "Dog Parks", "Amusement Parks", "Zoos", "Active Life"],
    "history": ["Landmarks & Historical Buildings", "Historical Tours", "Museums", "Walking Tours", "Tours"],
    "art": ["Art Galleries", "Museums", "Performing Arts", "Arts & Entertainment", "Art Classes"],
    "sports": ["Stadiums & Arenas", "Professional Sports Teams", "Sports Bars", "Golf", "Bowling", "Sports Clubs"],
}

# Budget -> target position on the cheapest(0)..priciest(1) city scale.
BUDGET_TARGETS = {"low": 0.0, "medium": 0.5, "high": 1.0}

# Global state
_MODEL = None
_EMBEDDING_MODEL = None
_CITY_TABLE = None
_CATEGORY_EMBEDDINGS = None

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

def _minmax(values):
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 1e-8 else np.full_like(values, 0.5)

def _category_embeddings(categories):
    """Embeddings for Yelp category names, used to match free-text interests."""
    global _CATEGORY_EMBEDDINGS
    if _CATEGORY_EMBEDDINGS is None:
        _, emb_model, _ = load_model()
        emb = emb_model.encode(list(categories), convert_to_numpy=True)
        _CATEGORY_EMBEDDINGS = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return _CATEGORY_EMBEDDINGS

def compute_activity_fit(activities, interests_text, category_share):
    """
    How well each city's mix of businesses fits the requested activities.

    Checkbox activities use the ACTIVITY_CATEGORIES mapping; free text is
    matched to the closest Yelp category names by embedding similarity.
    Each signal is min-max scaled across cities so it actually separates them.
    Returns an array aligned with category_share.index, or None if no input.
    """
    signals = []
    for activity in activities:
        cats = [c for c in ACTIVITY_CATEGORIES.get(activity.lower(), []) if c in category_share.columns]
        if cats:
            signals.append(_minmax(category_share[cats].sum(axis=1)))

    if interests_text and interests_text.strip():
        _, emb_model, _ = load_model()
        common = category_share.columns[category_share.sum() * len(category_share) >= 0.5]
        cat_emb = _category_embeddings(category_share.columns)
        text_emb = emb_model.encode(interests_text, convert_to_numpy=True)
        sims = cat_emb @ (text_emb / np.linalg.norm(text_emb))
        weights = pd.Series(np.clip(sims - 0.35, 0, None), index=category_share.columns)[common]
        if weights.sum() > 0:
            signals.append(_minmax(category_share[weights.index].values @ weights.values))

    if not signals:
        return None
    return np.mean(signals, axis=0)

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
        DataFrame with columns [city, rf_score, climate_match, activity_sim, budget_match, popularity, combined_score, ...]
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

    # Climate match: smooth falloff from the ideal temperature for the month
    dest_temp = city_table.get(f"dest_temp_month_{travel_month}", pd.Series(21.0, index=city_table.index))
    dest_precip = city_table.get(f"dest_precip_month_{travel_month}", pd.Series(0.0, index=city_table.index))
    climate_pref = user_prefs.get("climate_preference", "moderate").lower()
    target = CLIMATE_TARGETS_C.get(climate_pref, CLIMATE_TARGETS_C["moderate"])
    climate_match = np.exp(-((dest_temp.values - target) / CLIMATE_TOLERANCE_C) ** 2)

    # Build the RF feature matrix for all cities at once (one predict call)
    n = len(city_table)
    col = lambda name, default: city_table[name].values if name in city_table else np.full(n, default)
    features = {
        "user_avg_rating": np.full(n, user_avg_rating),
        "user_num_cities": np.full(n, user_num_cities),
        "user_num_reviews": np.full(n, user_num_reviews),
        "user_rating_std": np.full(n, user_rating_std),
        "dest_avg_rating": col("dest_avg_rating", 3.5),
        "dest_lat": col("dest_lat", 0),
        "dest_lon": col("dest_lon", 0),
        "dest_num_businesses": col("dest_num_businesses", 0),
        "dest_num_reviews": col("dest_num_reviews", 0),
        "dest_annual_avg_temp": col("dest_annual_avg_temp", 21),
        "dest_annual_precipitation": col("dest_annual_precipitation", 0),
        "dest_temp_range": col("dest_temp_range", 30),
    }
    for m in range(1, 13):
        features[f"dest_precip_month_{m}"] = col(f"dest_precip_month_{m}", 0)
        features[f"dest_temp_month_{m}"] = col(f"dest_temp_month_{m}", 21)
    dest_emb = np.stack([np.resize(np.asarray(e, dtype=np.float32), 384) for e in city_table["dest_embedding"]])
    for i in range(384):
        features[f"dest_emb_{i}"] = dest_emb[:, i]
        features[f"user_emb_{i}"] = np.full(n, user_embedding[i])

    # RF expects features in alphabetical order
    X = pd.DataFrame(features)[sorted(features)].values
    rf_scores = np.clip(rf_model.predict(X), 1, 5)  # expected rating 1-5

    results_df = pd.DataFrame({
        "city": city_table.index,
        "rf_score": rf_scores,
        "climate_match": climate_match,
        # Popularity: log of business count (review counts here are tiny and noisy)
        "popularity": np.log1p(col("dest_num_businesses", 0).astype(float)),
        "num_businesses": col("dest_num_businesses", 0).astype(int),
        "dest_temp": dest_temp.values.astype(float),
        "dest_precip": dest_precip.values.astype(float),
        "state": col("state", ""),
        "metro": col("metro", "") if "metro" in city_table else city_table.index.values,
    })
    results_df["popularity"] = _minmax(results_df["popularity"])
    # Blend absolute fit with fit relative to the other cities, so e.g. "warm in
    # January" still favors the warmest cities even when none reach the target.
    results_df["climate_match"] = 0.5 * results_df["climate_match"] + 0.5 * _minmax(results_df["climate_match"])

    try:
        from backend.city_table import get_city_profiles
    except ImportError:
        from city_table import get_city_profiles
    category_share, avg_price = get_city_profiles()
    cities = results_df["city"]

    activity_fit = compute_activity_fit(
        user_prefs.get("activities", []), interests_text, category_share.loc[cities]
    )
    results_df["activity_sim"] = activity_fit if activity_fit is not None else 0.5

    price_pos = _minmax(avg_price.loc[cities])
    budget_target = BUDGET_TARGETS.get(str(user_prefs.get("budget", "medium")).lower(), 0.5)
    results_df["budget_match"] = 1.0 - np.abs(price_pos - budget_target)

    # Normalize scores for combination
    results_df["rf_score_norm"] = (results_df["rf_score"] - results_df["rf_score"].min()) / (
        results_df["rf_score"].max() - results_df["rf_score"].min() + 1e-8
    )

    # Combined score: the user's stated preferences drive the ranking; the RF
    # rating prediction and popularity only break ties between similar fits.
    results_df["combined_score"] = (
        0.35 * results_df["climate_match"] +
        0.30 * results_df["activity_sim"] +
        0.10 * results_df["budget_match"] +
        0.15 * results_df["rf_score_norm"] +
        0.10 * results_df["popularity"]
    )

    results_df = results_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return results_df
