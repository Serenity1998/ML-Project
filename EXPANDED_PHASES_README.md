# 🚀 Expanded Dataset Pipeline (Phases 1-5)

Complete ML pipeline for destination recommendations with **expanded dataset** (500+ cities, 2M+ reviews).

## Overview

| Phase | Name | Output | Time |
|-------|------|--------|------|
| 1 | Data Loading | Filtered reviews (500+ cities) | 5 min |
| 2 | Destination Analysis | Destinations with stats | 1 min |
| 3 | Embeddings | 384D embeddings for all reviews | 10-30 min |
| 4 | Model Training | Trained Random Forest | 5 min |
| 5 | Ablation Study | Feature importance analysis | 5 min |

**Total Time: 30-45 minutes** (mostly embedding generation)

---

## Prerequisites

### Install Dependencies
```bash
pip install pandas numpy joblib scikit-learn sentence-transformers tqdm
```

### Files Needed
Ensure you have Yelp data in `raw_data/`:
- `yelp_academic_dataset_review.json` (6M+ reviews)
- `yelp_academic_dataset_business.json` (150k+ businesses)

### GPU (Optional but Recommended)
For faster embedding generation (3-5 min vs 30 min):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## How to Run

### Option A: Run All Phases Sequentially
```bash
python phase1_expanded.py
python phase2_expanded.py
python phase3_expanded.py  # ⏳ Takes 10-30 minutes
python phase4_expanded.py
python phase5_expanded.py
```

### Option B: Run Individual Phases
Each phase reads outputs from the previous phase:
```bash
# Start from any point
python phase2_expanded.py  # Requires Phase 1 output
python phase3_expanded.py  # Requires Phases 1-2 output
```

### Option C: Quick Test (Small Dataset)
To test the pipeline quickly:
```python
# In phase2_expanded.py, add after loading:
reviews_df = reviews_df.sample(n=5000, random_state=42)  # Test with 5k reviews
```

---

## What Each Phase Does

### Phase 1: Data Loading
- Loads full Yelp review/business data
- Filters to 500+ cities (half of all cities)
- Filters to cross-city users (2+ cities)
- **Output:** `reviews_phase1_expanded.parquet`

### Phase 2: Destination Analysis
- Analyzes each destination (city)
- Filters to cities with 50+ reviews
- Creates destination statistics
- **Output:** `destinations_phase2_expanded.parquet`

### Phase 3: Embeddings
- Generates 384D Sentence-BERT embeddings for all reviews
- Creates city embeddings (average of review embeddings)
- ⏳ **Slowest step** (uses GPU if available)
- **Outputs:** 
  - `review_embeddings_phase3_expanded.npy` (2M embeddings)
  - `destination_embeddings_phase3_expanded.joblib` (500+ cities)

### Phase 4: Model Training
- Builds feature matrix from reviews + embeddings
- Trains Random Forest model (normalized features)
- Evaluates on test set
- **Output:** `random_forest_model_phase4_expanded.joblib`

### Phase 5: Ablation Study
- Analyzes which features matter most
- Shows feature importance by category
- Validates model design choices
- **Output:** Feature importance breakdown

---

## Expected Results

```
PHASE 1:
  Total reviews: 2,000,000+
  Unique users: 500,000+
  Unique cities: 500+

PHASE 4:
  MSE:  ~0.95 (better than original 1.14)
  MAE:  ~0.75 (better than original 0.88)
  R²:   ~0.15 (improved from -0.21)

PHASE 5:
  Embeddings:  68-75% of model power
  Ratings:     20-25% of model power
  Weather:     3-5% of model power
```

---

## Troubleshooting

### Phase 3 Takes Too Long
**Problem:** Embedding generation taking 30+ minutes

**Solutions:**
1. **Enable GPU** (if not already):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   # If False, install CUDA version of PyTorch
   ```

2. **Reduce batch size** (in phase3_expanded.py):
   ```python
   batch_size = 64  # Change from 128 to 64
   ```

3. **Use subset of data** (test run):
   ```python
   reviews_df = reviews_df.sample(frac=0.1, random_state=42)  # 10% of data
   ```

### Missing Data Files
```
FileNotFoundError: raw_data/yelp_academic_dataset_review.json

→ Download from: https://www.yelp.com/dataset
→ Place in: raw_data/ directory
```

### Out of Memory
**Problem:** "MemoryError" during embedding generation

**Solutions:**
1. Reduce batch size (in phase3_expanded.py):
   ```python
   batch_size = 32
   ```

2. Process in chunks (modify phase3_expanded.py):
   ```python
   reviews_df = reviews_df.iloc[:100000]  # First 100k reviews
   ```

---

## Next Steps

After completing all phases:

1. **Update frontend/backend:**
   ```bash
   # Update app.py to use new embeddings
   # Restart Flask server
   python app.py
   ```

2. **Test with 500+ cities:**
   - Open `index.html` in browser
   - Select preferences
   - Get recommendations from 500+ cities!

3. **For teammates:**
   - Share this README
   - Run phases in order
   - Results are reproducible and self-contained

---

## File Outputs

```
cache/
  ├── reviews_phase1_expanded.parquet
  ├── reviews_phase2_expanded.parquet
  ├── destinations_phase2_expanded.parquet
  ├── review_embeddings_phase3_expanded.npy
  └── destination_embeddings_phase3_expanded.joblib

outputs/
  └── random_forest_model_phase4_expanded.joblib

reports/
  ├── phase4_expanded_results.txt
  └── phase5_expanded_ablation.txt
```

---

## Questions?

Each phase prints detailed progress output. Check console for:
- ✓ Data loaded successfully
- ✓ Sizes of datasets
- ✓ Model performance metrics
- ✓ File paths for outputs

Happy ML! 🚀
