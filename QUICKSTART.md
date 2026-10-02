# Quick Start Guide — Phase 1

Get the destination recommendation project running in 6 steps.

## Prerequisites

- Python 3.8+
- Yelp Academic Dataset (5GB) downloaded to `data/raw/`
- ~2-3 hours for Phase 1 to complete

## 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Verify Data Files

Place these in `data/raw/`:
- `yelp_academic_dataset_business.json` (~130MB)
- `yelp_academic_dataset_review.json` (~5.5GB)
- `yelp_academic_dataset_user.json` (~280MB)

```bash
ls data/raw/yelp*.json
```

## 3. Run Phase 1 Notebooks

Open Jupyter and run notebooks in order:

```bash
cd notebooks/phase1_data_prep

# 1️⃣ Load 10% sample of Yelp data
jupyter notebook 01_data_loading.ipynb

# 2️⃣ Filter to major cities (50+ businesses, 500+ reviews)
jupyter notebook 02_destination_filtering.ipynb

# 3️⃣ Identify users who reviewed in 2+ cities
jupyter notebook 03_cross_city_users.ipynb

# 4️⃣ Extract user/destination features
jupyter notebook 04_feature_engineering.ipynb

# 5️⃣ Set up train/test split and evaluation
jupyter notebook 05_evaluation_harness.ipynb

# 6️⃣ Train and evaluate baselines
jupyter notebook 06_baselines.ipynb
```

## 4. Expected Outputs

After Phase 1, you'll have:

**In `data/cache/`:**
- `businesses_sample.parquet` — 10% sample of businesses
- `reviews_sample.parquet` — 10% sample of reviews
- `businesses_major_cities.parquet` — filtered to major cities
- `reviews_cross_city.parquet` — multi-city user reviews
- `train_reviews.parquet`, `test_reviews.parquet` — train/test split
- Metadata files with statistics

**In `outputs/phase1/`:**
- Baseline model results (Popularity, Cosine Similarity)
- Evaluation metrics (Recall@10, NDCG@10, MRR)
- Comparison tables

## 5. Key Configuration (config.py)

Adjust these for different data sizes:

```python
PHASE1_SAMPLE_FRACTION = 0.1          # 10% = ~500MB (fast)
# Increase to 0.3 for ~1.5GB if you have more RAM

PHASE1_MIN_BUSINESSES_PER_CITY = 50   # Cities with ≥50 businesses
PHASE1_MIN_USER_CITIES = 2            # Users who visited 2+ cities
PHASE1_MIN_USER_REVIEWS = 10          # Users with ≥10 reviews
```

## 6. Troubleshooting

### Out of Memory
If you run out of RAM during sampling:
- Reduce `PHASE1_SAMPLE_FRACTION` to 0.05
- Close other applications
- Restart kernel between notebooks

### Slow Loading
- First run: slow (reads 5GB JSON)
- Subsequent runs: fast (uses cached parquet)
- Check `data/cache/` for cached files

### Missing Data
```bash
# Verify Yelp files exist
ls -lh data/raw/yelp*.json

# They should be:
# business.json:  ~130 MB
# review.json:    ~5.5 GB
# user.json:      ~280 MB
```

## What's Next

After Phase 1 ✓:
- **Phase 2** (weeks 7-8): Add weather features from Open-Meteo
- **Phase 3** (week 9): Generate review embeddings with Sentence-BERT
- **Phase 4** (week 10): Train Random Forest + Two-Tower neural network
- **Phase 5** (week 11): Ablation study (feature combinations)
- **Phase 6** (week 12): Results analysis and visualization

## Questions?

Check `README.md` for full project details and references.
