"""
Rerun the ranking experiment (notebooks/phase4_models/03_ranking_models.ipynb) with an optional
minimum trip distance: held-out, negative and candidate cities must be at least --min-distance km
from the user's home city, so "trips" are real travel instead of suburb hops.

    python scripts/ranking_by_trip_distance.py --min-distance 0    # original setting
    python scripts/ranking_by_trip_distance.py --min-distance 50   # real trips only

The first run rebuilds the Phase 1 data (10% user sample) into data/cache/ranking_sample/,
leaving the app's caches in data/cache/ untouched.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import *
from src.data.cache import DataCache
from src.data.loader import load_businesses, load_reviews_by_user_sample
from src.data.normalizer import normalize_city_names, get_major_cities, filter_to_major_cities
from src.evaluation.harness import RecommendationHarness
from src.features.ranking_features import (
    USER_FEATURES, CITY_FEATURES, PAIR_FEATURES, WEATHER_FEATURES, TEXT_FEATURES,
    city_features, user_features, trip_months, city_month_weather, build_pair_features,
)
from src.features.review_embeddings import user_text_vectors, city_text_vectors, text_features
from src.features.weather_features import WeatherFeatureBuilder
from src.models.baselines import PopularityBaseline
from src.models.rankers import RandomForestRanker, TwoTowerRanker

cache = DataCache(str(CACHE_DIR / 'ranking_sample'))


def load_data():
    """Phase 1 notebooks 01-03: 10% of users, major cities, users with 2+ cities and 10+ reviews."""
    businesses, reviews = cache.load_dataframe('businesses_major_cities'), cache.load_dataframe('reviews_cross_city')
    if businesses is not None and reviews is not None:
        return businesses, reviews
    print('Building the 10% user sample from raw Yelp data (one-time, slow)...', flush=True)
    businesses = normalize_city_names(load_businesses(BUSINESS_FILE), city_column='city')
    reviews = load_reviews_by_user_sample(REVIEW_FILE, PHASE1_USER_SAMPLE_FRACTION, PHASE1_SAMPLE_SEED)

    major = get_major_cities(businesses, PHASE1_MIN_BUSINESSES_PER_CITY, PHASE1_MIN_REVIEWS_PER_CITY)
    businesses = filter_to_major_cities(businesses, 'city', major)
    reviews = reviews[reviews['business_id'].isin(businesses['business_id'])]

    merged = reviews.merge(businesses[['business_id', 'city']], on='business_id')
    multi_city = merged.groupby('user_id')['city'].nunique()
    multi_city = multi_city[multi_city >= PHASE1_MIN_USER_CITIES].index
    counts = reviews[reviews['user_id'].isin(multi_city)]['user_id'].value_counts()
    reviews = reviews[reviews['user_id'].isin(counts[counts >= PHASE1_MIN_USER_REVIEWS].index)]

    cache.save_dataframe(businesses, 'businesses_major_cities')
    cache.save_dataframe(reviews, 'reviews_cross_city')
    return businesses, reviews


def run(min_distance_km):
    businesses, reviews = load_data()
    harness = RecommendationHarness(
        reviews, businesses, test_size=PHASE1_TEST_SIZE, val_size=PHASE1_VAL_SIZE,
        random_state=RANDOM_STATE, n_negatives=PHASE1_NEGATIVES_PER_POSITIVE, min_distance_km=min_distance_km,
    )
    train_reviews, test_reviews = harness.create_train_test_split()
    all_cities = sorted(businesses['city'].unique())
    train_labels = harness.get_training_labels(all_cities)
    val_pairs, test_pairs = harness.get_validation_pairs(), harness.get_test_pairs()
    print(f"Min distance {min_distance_km} km · users: train {len(harness.train_users):,} · "
          f"val {len(harness.val_users):,} · test {len(harness.test_users):,}", flush=True)

    history = pd.concat([train_reviews, harness.val_reviews, test_reviews])
    cities = city_features(businesses)
    users = user_features(history, businesses, cities)
    months = trip_months(reviews, businesses, harness.targets)
    weather = city_month_weather(cities, WeatherFeatureBuilder(cache_dir=WEATHER_CACHE))
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    text = text_features(user_text_vectors(history, encoder), city_text_vectors(history, businesses, encoder))
    train_df = build_pair_features(train_labels, users, cities, months, weather, text)
    print('Features built', flush=True)

    def scorer(model):
        return lambda candidates: model.score(build_pair_features(candidates, users, cities, months, weather, text))

    def evaluate(score_fn, pairs=test_pairs, per_user=False):
        return harness.evaluate(pairs, all_cities, score_fn, k=PHASE1_EVAL_K, per_user=per_user)

    recall = f'recall@{PHASE1_EVAL_K}'
    popularity = PopularityBaseline(businesses, history)
    popularity_scores = {city: popularity.score_city(city) for city in all_cities}
    rng = np.random.default_rng(RANDOM_STATE)
    results = {
        'Random': evaluate(lambda c: rng.random(len(c))),
        'Popularity': evaluate(lambda c: c['city'].map(popularity_scores).to_numpy()),
    }

    base = USER_FEATURES + CITY_FEATURES + PAIR_FEATURES
    user_pcs, city_pcs = list(text['user_pcs'].columns), list(text['city_pcs'].columns)
    city_weather = ['city_temp', 'city_precip']
    configs = {
        'Baseline': (base, USER_FEATURES, CITY_FEATURES),
        '+Weather': (base + WEATHER_FEATURES, USER_FEATURES, CITY_FEATURES + city_weather),
        '+Embeddings': (base + TEXT_FEATURES, USER_FEATURES + user_pcs, CITY_FEATURES + city_pcs),
        'Full': (base + WEATHER_FEATURES + TEXT_FEATURES, USER_FEATURES + user_pcs,
                 CITY_FEATURES + city_weather + city_pcs),
    }

    val_by_depth = {}
    for depth in [8, 15]:
        rf = RandomForestRanker(base, RF_N_ESTIMATORS, depth, RF_RANDOM_STATE).fit(train_df)
        val_by_depth[depth] = evaluate(scorer(rf), val_pairs)[recall]
    best_depth = max(val_by_depth, key=val_by_depth.get)

    importance = {}
    for name, (rf_cols, _, _) in configs.items():
        rf = RandomForestRanker(rf_cols, RF_N_ESTIMATORS, best_depth, RF_RANDOM_STATE).fit(train_df)
        results[f'Random Forest · {name}'] = evaluate(scorer(rf))
        importance[name] = rf.feature_importance()
        print(f'Random Forest · {name} done', flush=True)

    val_fn = lambda model: evaluate(scorer(model), val_pairs)[recall]
    for name, (_, user_cols, city_cols) in configs.items():
        tt = TwoTowerRanker(
            user_cols, city_cols, TOWER_HIDDEN_DIM, TOWER_EMBEDDING_DIM, TOWER_LEARNING_RATE, epochs=TOWER_EPOCHS
        ).fit(train_df, val_fn)
        results[f'Two-Tower · {name}'] = evaluate(scorer(tt))
        print(f'Two-Tower · {name} done', flush=True)

    table = pd.DataFrame(results).T.round(3)
    top_features = pd.Series(importance['Full']).sort_values(ascending=False).head(8).round(3)
    print(f'\nTest users: {len(test_pairs):,} · RF depth {best_depth}\n')
    print(table.to_string())
    print('\nRandom Forest · Full feature importance:\n' + top_features.to_string())

    out = REPORT_DIR / f'ranking_min_distance_{min_distance_km}km.json'
    out.write_text(json.dumps({
        'min_distance_km': min_distance_km,
        'test_users': len(test_pairs),
        'rf_depth': best_depth,
        'results': table.to_dict(orient='index'),
        'rf_full_importance': top_features.to_dict(),
    }, indent=2), encoding='utf-8')
    print(f'\nSaved {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-distance', type=float, default=0)
    args = parser.parse_args()
    run(int(args.min_distance) if args.min_distance.is_integer() else args.min_distance)
