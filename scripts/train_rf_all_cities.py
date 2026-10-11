"""
Retrain the app's rating Random Forest on reviews from every city in the city
table (~476), not just the 39 big ones random_forest_model_expanded saw.

Same 804 features as before, so backend/scorer.py can load it unchanged:
  user stats (4)      rating history from the user's OTHER reviews (leave-one-out,
                      so the review being predicted never leaks into its features)
  user_emb (384)      mean embedding of up to USER_HISTORY_TEXTS other reviews
  dest scalars        straight from city_table_v2, exactly what the app feeds
  dest_emb (384)      mean embedding of training-split reviews in the city

Train/test is split by user, so no test user's ratings shape the model or the
city stats. Steps:
  1. scan reviews: per-user aggregates over all reviews, pick up to
     REVIEWS_PER_CITY target reviews per city                       (~10 min)
  2. scan again: collect target texts and user history texts       (~10 min)
  3. embed on GPU, build features, train, evaluate against baselines
  4. save model, metrics, and the new city stats/embeddings

Outputs:
  outputs/random_forest_model_all_cities.joblib
  data/cache/city_training_stats_all.parquet   (read by build_city_table.py)
  data/cache/city_table_v2.parquet              (patched in place)
  reports/all_cities_rf_results.txt

Run from the project root:  python scripts/train_rf_all_cities.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_city_table import CACHE_DIR, OUTPUT_DIR, RAW_BUSINESS, apply_city_stats, normalize_city  # noqa: E402
from src.models.random_forest import RandomForestRecommender  # noqa: E402

RAW_REVIEW = PROJECT_ROOT / "data" / "raw" / "yelp_academic_dataset_review.json"
SAMPLE_CACHE = CACHE_DIR / "rf_all_cities_sample.parquet"
HISTORY_CACHE = CACHE_DIR / "rf_all_cities_history.parquet"
CITY_STATS = CACHE_DIR / "city_training_stats_all.parquet"
MODEL_PATH = OUTPUT_DIR / "random_forest_model_all_cities.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "all_cities_rf_results.txt"

REVIEWS_PER_CITY = 500      # target reviews per city; caps the big metros
USER_HISTORY_TEXTS = 5      # other reviews averaged into a user's embedding
CITY_TEXTS = 200            # training reviews averaged into a city's embedding
TEXT_CHARS = 1000
TEST_USER_FRACTION = 0.2
SEED = 42

# A user with no other reviews looks like the app's cold-start user.
NO_HISTORY = {"user_avg_rating": 3.5, "user_rating_std": 1.0, "user_num_reviews": 0.0, "user_num_cities": 0.0}

DEST_SCALARS = (
    ["dest_lat", "dest_lon", "dest_num_businesses", "dest_annual_avg_temp",
     "dest_annual_precipitation", "dest_temp_range"]
    + [f"dest_temp_month_{m}" for m in range(1, 13)]
    + [f"dest_precip_month_{m}" for m in range(1, 13)]
)


def business_cities(city_table):
    """business_id -> (any-city key for counting user cities, table row index or -1)."""
    b = pd.read_json(RAW_BUSINESS, lines=True)[["business_id", "city", "state"]]
    keys = b["city"].map(normalize_city) + ", " + b["state"]
    key_codes, _ = pd.factorize(keys)
    table_pos = pd.Index(city_table.index).get_indexer(keys)
    return dict(zip(b["business_id"], zip(key_codes, table_pos)))


def scan_reviews(biz):
    """Pass 1: one slim row per review (line number, user, city, stars)."""
    user_codes, rows = {}, []
    with open(RAW_REVIEW, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            r = json.loads(line)
            any_city, table_city = biz.get(r["business_id"], (-1, -1))
            user = user_codes.setdefault(r["user_id"], len(user_codes))
            rows.append((line_no, user, any_city, table_city, r["stars"]))
            if line_no % 1_000_000 == 0:
                print(f"      pass 1: {line_no:,} reviews")
    return pd.DataFrame(rows, columns=["line", "user", "any_city", "city_pos", "stars"])


def leave_one_out_user_stats(all_reviews, targets):
    """Each target's user stats from that user's other reviews only."""
    s = all_reviews["stars"].astype(float)
    per_user = all_reviews.assign(s=s, s2=s ** 2).groupby("user").agg(
        n=("s", "size"), total=("s", "sum"), total2=("s2", "sum"), cities=("any_city", "nunique")
    )
    same_city = all_reviews.groupby(["user", "any_city"]).size().rename("n_same_city")
    t = targets.join(per_user, on="user").join(same_city, on=["user", "any_city"])

    n = t["n"] - 1
    star = t["stars"].astype(float)
    mean = (t["total"] - star) / n.where(n > 0)
    var = (t["total2"] - star ** 2) / n.where(n > 0) - mean ** 2
    out = pd.DataFrame({
        "user_avg_rating": mean,
        "user_rating_std": np.sqrt(var.clip(lower=0)),
        "user_num_reviews": n.astype(float),
        "user_num_cities": (t["cities"] - (t["n_same_city"] == 1)).astype(float),
    }, index=t.index)
    no_history = n == 0
    for col, value in NO_HISTORY.items():
        out.loc[no_history, col] = value
    return out


def collect_texts(targets, all_reviews):
    """Pass 2: texts of the target reviews and of each target user's other reviews."""
    target_lines = set(targets["line"])
    target_users = set(targets["user"])
    rng = np.random.default_rng(SEED)
    # Up to USER_HISTORY_TEXTS + 1 reviews per target user, so one can be dropped
    # when it is the target itself.
    pool = all_reviews[all_reviews["user"].isin(target_users)]
    pool = pool.iloc[rng.permutation(len(pool))].groupby("user").head(USER_HISTORY_TEXTS + 1)
    history_lines = dict(zip(pool["line"], pool["user"]))

    texts, history = {}, []
    with open(RAW_REVIEW, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no in target_lines or line_no in history_lines:
                text = json.loads(line)["text"][:TEXT_CHARS]
                if line_no in target_lines:
                    texts[line_no] = text
                if line_no in history_lines:
                    history.append((history_lines[line_no], line_no, text))
            if line_no % 1_000_000 == 0:
                print(f"      pass 2: {line_no:,} reviews")
    targets = targets.assign(text=targets["line"].map(texts))
    return targets, pd.DataFrame(history, columns=["user", "line", "text"])


def build_sample(city_table):
    if SAMPLE_CACHE.exists() and HISTORY_CACHE.exists():
        print("[1-2/4] Using cached review sample")
        return pd.read_parquet(SAMPLE_CACHE), pd.read_parquet(HISTORY_CACHE)

    print("[1/4] Scanning reviews (pass 1)...")
    biz = business_cities(city_table)
    all_reviews = scan_reviews(biz)
    print(f"      {len(all_reviews):,} reviews, {all_reviews['user'].nunique():,} users")

    in_table = all_reviews[all_reviews["city_pos"] >= 0]
    targets = in_table.sample(frac=1, random_state=SEED).groupby("city_pos").head(REVIEWS_PER_CITY)
    targets = targets.join(leave_one_out_user_stats(all_reviews, targets))
    print(f"      {len(targets):,} target reviews across {targets['city_pos'].nunique()} cities")

    print("[2/4] Collecting review texts (pass 2)...")
    targets, history = collect_texts(targets, all_reviews)
    targets.to_parquet(SAMPLE_CACHE)
    history.to_parquet(HISTORY_CACHE)
    return targets, history


def embed(texts, model):
    return model.encode(list(texts), batch_size=256, convert_to_numpy=True,
                        show_progress_bar=True).astype(np.float32)


def user_embeddings(targets, history, model):
    """Normalized mean embedding of each target user's other reviews (zeros if none)."""
    history_emb = embed(history["text"], model)
    by_user = {user: (rows["line"].to_numpy(), history_emb[rows.index])
               for user, rows in history.reset_index(drop=True).groupby("user")}
    result = np.zeros((len(targets), 384), dtype=np.float32)
    for i, (user, line) in enumerate(zip(targets["user"], targets["line"])):
        if user not in by_user:
            continue
        lines, embs = by_user[user]
        embs = embs[lines != line][:USER_HISTORY_TEXTS]
        if len(embs):
            mean = embs.mean(axis=0)
            result[i] = mean / (np.linalg.norm(mean) + 1e-8)
    return result


def city_stats(train, target_emb, city_table):
    """Rating, review count, and embedding per city from training-split reviews only."""
    g = train.groupby("city_pos")
    stats = pd.DataFrame({
        "dest_avg_rating": g["stars"].mean(),
        "dest_num_reviews": g.size(),
    })
    # train's index labels are row positions in targets (and so in target_emb)
    rng = np.random.default_rng(SEED)
    stats["dest_embedding"] = [
        target_emb[rng.permutation(rows.index.to_numpy())[:CITY_TEXTS]].mean(axis=0).tolist()
        for _, rows in g
    ]
    stats.index = city_table.index[stats.index]
    return stats


def feature_matrix(rows, user_emb, stats, city_table):
    cities = city_table.index[rows["city_pos"]]
    cols = {c: rows[c].to_numpy(float) for c in NO_HISTORY}
    # A city whose reviews all went to test users falls back to its Yelp stars and the mean embedding
    cols["dest_avg_rating"] = stats["dest_avg_rating"].reindex(cities).fillna(
        city_table["yelp_avg_stars"].reindex(cities)).to_numpy(float)
    cols["dest_num_reviews"] = stats["dest_num_reviews"].reindex(cities).fillna(0).to_numpy(float)
    for c in DEST_SCALARS:
        cols[c] = city_table[c].reindex(cities).to_numpy(float)
    mean_emb = np.mean(np.stack(stats["dest_embedding"].map(np.asarray)), axis=0)
    dest_emb = np.stack([mean_emb if not isinstance(e, list) else np.asarray(e)
                         for e in stats["dest_embedding"].reindex(cities)])
    for i in range(384):
        cols[f"dest_emb_{i}"] = dest_emb[:, i]
        cols[f"user_emb_{i}"] = user_emb[:, i]
    names = sorted(cols)
    return pd.DataFrame(cols)[names].to_numpy(np.float32), names


def main():
    city_table = pd.read_parquet(CACHE_DIR / "city_table_v2.parquet")
    missing = [c for c in DEST_SCALARS if c not in city_table]
    assert not missing, f"city_table_v2 is missing {missing}"

    targets, history = build_sample(city_table)
    targets = targets.reset_index(drop=True)

    print("[3/4] Embedding texts and training...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    target_emb = embed(targets["text"], model)
    user_emb = user_embeddings(targets, history, model)

    users = targets["user"].unique()
    test_users = set(np.random.default_rng(SEED).choice(users, int(len(users) * TEST_USER_FRACTION), replace=False))
    is_test = targets["user"].isin(test_users).to_numpy()
    train, test = targets[~is_test], targets[is_test]
    stats = city_stats(train, target_emb, city_table)

    X_train, names = feature_matrix(train, user_emb[~is_test], stats, city_table)
    X_test, _ = feature_matrix(test, user_emb[is_test], stats, city_table)
    y_train, y_test = train["stars"].to_numpy(float), test["stars"].to_numpy(float)
    print(f"      train {len(train):,} reviews / test {len(test):,} reviews (split by user)")

    rf = RandomForestRecommender(n_estimators=100, max_depth=15, random_state=SEED)
    rf.feature_names = names
    rf.model = RandomForestRegressor(n_estimators=100, max_depth=15, max_features=0.33,
                                     min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    rf.model.fit(rf.scaler.fit_transform(X_train), y_train)
    rf.feature_importance_ = rf.model.feature_importances_

    has_history = (test["user_num_reviews"] > 0).to_numpy()
    preds = {
        "Random Forest (all cities)": rf.predict(X_test),
        "Baseline: global mean": np.full(len(test), y_train.mean()),
        "Baseline: city mean": stats["dest_avg_rating"].reindex(city_table.index[test["city_pos"]])
                                    .fillna(y_train.mean()).to_numpy(),
        "Baseline: user's own average": test["user_avg_rating"].to_numpy(),
    }
    lines = [f"Train: {len(train):,} reviews, {len(set(train['user'])):,} users | "
             f"Test: {len(test):,} reviews, {len(test_users):,} users | "
             f"{train['city_pos'].nunique()} cities | split by user, leave-one-out user stats", ""]
    for label, mask in [("All test reviews", np.ones(len(test), bool)),
                        (f"Test users with history ({has_history.sum():,})", has_history)]:
        lines.append(label)
        for name, p in preds.items():
            p, y = p[mask], y_test[mask]
            lines.append(f"  {name:<30} RMSE {mean_squared_error(y, p) ** 0.5:.4f}  "
                         f"MAE {mean_absolute_error(y, p):.4f}  R2 {r2_score(y, p):+.4f}")
        lines.append("")
    top = pd.Series(rf.feature_importance_, index=names)
    groups = {"user stats": top[list(NO_HISTORY)].sum(), "user_emb": top.filter(like="user_emb").sum(),
              "dest_emb": top.filter(like="dest_emb").sum()}
    groups["dest scalars"] = 1 - sum(groups.values())
    lines.append("Feature importance by group: " + ", ".join(f"{k} {v:.1%}" for k, v in groups.items()))
    lines.append("Top features: " + ", ".join(f"{k} {v:.3f}" for k, v in top.nlargest(8).items()))
    report = "\n".join(lines)
    print(report)

    print("[4/4] Saving...")
    joblib.dump(rf, MODEL_PATH)
    stats.to_parquet(CITY_STATS)
    apply_city_stats(city_table, stats).to_parquet(CACHE_DIR / "city_table_v2.parquet")
    REPORT_PATH.write_text("ALL-CITIES RANDOM FOREST\n" + "=" * 70 + "\n" + report + "\n", encoding="utf-8")
    print(f"      wrote {MODEL_PATH.name}, {CITY_STATS.name}, patched city_table_v2, {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
