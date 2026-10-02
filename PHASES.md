# Project Phases — Detailed Breakdown

## Phase 1: Data Preparation & Baselines (Weeks 1-6)

**Goal:** Verify the recommendation problem can be constructed successfully.

### Tasks
- ✅ Load and sample Yelp data (10% of 5GB)
- ✅ Normalize city names
- ✅ Identify major cities (≥50 businesses, ≥500 reviews)
- ✅ Identify cross-city users (reviewed in ≥2 cities)
- ✅ Extract user preferences (rating behavior, categories, price)
- ✅ Extract destination characteristics (ratings, reviews, categories)
- ✅ Create train/test split (by user, not review)
- ✅ Implement evaluation harness (Recall@10, NDCG@10, MRR)
- ✅ Implement Popularity baseline
- ✅ Implement Cosine Similarity baseline
- ✅ Generate baseline results on evaluation harness

### Notebooks
```
notebooks/phase1_data_prep/
├── 01_data_loading.ipynb             ← Load sample (10%)
├── 02_destination_filtering.ipynb    ← Filter to major cities
├── 03_cross_city_users.ipynb         ← Identify multi-city users
├── 04_feature_engineering.ipynb      ← Build user/destination profiles
├── 05_evaluation_harness.ipynb       ← Train/test split, metrics
└── 06_baselines.ipynb                ← Popularity + Cosine Similarity
```

### Deliverable
✅ Baseline results confirming that personalized recommendations beat popularity

---

## Phase 2: Weather Features (Weeks 7-8)

**Goal:** Add month-specific climate data to destination features.

### Tasks
- [ ] Fetch city coordinates (geolocation lookup)
- [ ] Query Open-Meteo API for historical weather
- [ ] Extract monthly temperature & rainfall for each city
- [ ] Compute features: avg temp, rainfall, deviation from user's home climate
- [ ] Join weather with destination profiles
- [ ] Re-evaluate baselines with weather features
- [ ] Measure improvement over Phase 1 baselines

### Notebooks
```
notebooks/phase2_weather/
└── 01_weather_features.ipynb        ← Weather feature engineering
```

### Deliverable
📊 Comparison: Do weather features improve recommendation quality?

---

## Phase 3: Review Embeddings (Week 9)

**Goal:** Extract semantic information from review text using Sentence-BERT.

### Tasks
- [ ] Sample reviews (1000 per city, limit computation)
- [ ] Clean review text (lowercase, remove URLs/special chars)
- [ ] Load Sentence-BERT model (`all-MiniLM-L6-v2`)
- [ ] Generate embeddings for sampled reviews
- [ ] Cache embeddings for later use
- [ ] Aggregate embeddings to user profiles (mean)
- [ ] Aggregate embeddings to destination profiles (mean)
- [ ] Add embedding dimensions to feature vectors

### Notebooks
```
notebooks/phase3_embeddings/
└── 01_review_embeddings.ipynb       ← Generate & cache embeddings
```

### Deliverable
✅ Cached embeddings; user/destination feature vectors with semantic info

---

## Phase 4: Machine Learning Models (Week 10)

**Goal:** Train supervised models beyond baselines.

### 4a. Random Forest

**Tasks**
- [ ] Prepare training data: (user_features + dest_features) → compatibility
- [ ] Handle mixed feature types (continuous, categorical)
- [ ] Train Random Forest classifier/regressor
- [ ] Tune hyperparameters: n_estimators, max_depth, etc.
- [ ] Generate predictions for test set
- [ ] Compute Recall@10, NDCG@10, MRR

### 4b. Two-Tower Neural Network

**Architecture**
```
User Features ──→ [UserTower] ──→ User Embedding (64d)
                                         ↓
                                    Dot Product  ↓  Compatibility Score
                                         ↑
Dest Features ──→ [DestTower] ──→ Dest Embedding (64d)
```

**Tasks**
- [ ] Define user and destination tower networks (MLPs)
- [ ] Implement ranking loss (e.g., triplet loss or BPR)
- [ ] Train on user-destination pairs
- [ ] Tune: embedding_dim, learning_rate, batch_size, epochs
- [ ] Generate predictions for test set
- [ ] Compute evaluation metrics

### Notebooks
```
notebooks/phase4_models/
├── 01_random_forest.ipynb           ← RF model training
└── 02_two_tower.ipynb               ← Neural network training
```

### Deliverable
🤖 Two trained models; evaluation results on test set

---

## Phase 5: Ablation Experiments (Week 11)

**Goal:** Measure the contribution of each feature component.

### Feature Configurations to Test

| Config | Features | Description |
|--------|----------|-------------|
| **Baseline** | User rating behavior + Destination stats | No weather, no embeddings |
| **+Weather** | + Monthly weather features | Test weather contribution |
| **+Embeddings** | + Review embeddings | Test semantic info contribution |
| **Full** | + Both weather and embeddings | Best combination |

### Tasks
- [ ] Prepare feature matrices for each config
- [ ] Train Random Forest for each config
- [ ] Train Two-Tower network for each config
- [ ] Evaluate all combinations on test set
- [ ] Create results table:

| Model | Config | Recall@10 | NDCG@10 | MRR |
|-------|--------|-----------|---------|-----|
| Popularity | — | ? | ? | ? |
| Cosine Sim | — | ? | ? | ? |
| Random Forest | Baseline | ? | ? | ? |
| Random Forest | +Weather | ? | ? | ? |
| ... | ... | ... | ... | ... |
| Two-Tower | Full | ? | ? | ? |

### Notebooks
```
notebooks/phase5_ablation/
└── 01_ablation_study.ipynb          ← Run all feature combinations
```

### Deliverable
📊 Ablation results table showing which features matter most

---

## Phase 6: Analysis & Presentation (Week 12)

**Goal:** Visualize, interpret, and present results.

### Tasks
- [ ] Aggregate all results into comparison tables
- [ ] Create visualizations:
  - Bar chart: Recall@10 across all models/configs
  - Bar chart: NDCG@10 across all models/configs
  - Feature importance (Random Forest + permutation importance)
  - Error analysis: Which user/city pairs fail?
- [ ] Permutation feature importance analysis
- [ ] Error analysis: Failed recommendations (why?)
- [ ] Example recommendations: Show top-5 for sample users
- [ ] Hypothesis tests: Is improvement statistically significant?

### Notebooks
```
notebooks/phase6_analysis/
└── 01_results_analysis.ipynb        ← Visualization & analysis
```

### Deliverable
📈 Final report with tables, figures, and interpretation

---

## Summary

| Phase | Duration | Key Outcome | Dependencies |
|-------|----------|-------------|--------------|
| 1 | Weeks 1-6 | Working evaluation harness + baselines | Yelp data |
| 2 | Weeks 7-8 | Weather features | Phase 1 ✓ |
| 3 | Week 9 | Review embeddings | Phase 1 ✓ |
| 4 | Week 10 | Trained ML models | Phase 1-3 ✓ |
| 5 | Week 11 | Feature ablation | Phase 4 ✓ |
| 6 | Week 12 | Results analysis | Phase 5 ✓ |

**Total: 12 weeks**
