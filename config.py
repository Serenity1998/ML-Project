from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "samples"
CACHE_DIR = DATA_DIR / "cache"

# Output directories
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"

# Yelp files
BUSINESS_FILE = RAW_DATA_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = RAW_DATA_DIR / "yelp_academic_dataset_review.json"
USER_FILE = RAW_DATA_DIR / "yelp_academic_dataset_user.json"
TIPS_FILE = RAW_DATA_DIR / "yelp_academic_dataset_tip.json"

# Reproducibility
RANDOM_STATE = 42

# ============================================================================
# PHASE 1: Data Preparation & Baselines (Weeks 1-6)
# ============================================================================

# Data sampling for Phase 1 (work with smaller subset)
PHASE1_SAMPLE_FRACTION = 0.1  # 10% sample for development
PHASE1_SAMPLE_SEED = 42

# Destination filtering for Phase 1
PHASE1_MIN_BUSINESSES_PER_CITY = 50
PHASE1_MIN_REVIEWS_PER_CITY = 500

# User filtering for Phase 1
PHASE1_MIN_USER_REVIEWS = 10
PHASE1_MIN_USER_CITIES = 2  # Cross-city requirement

# Evaluation for Phase 1
PHASE1_TEST_SIZE = 0.2
PHASE1_EVAL_K = 10  # Compute Recall@10, NDCG@10, MRR@10

# ============================================================================
# PHASE 2: Weather Features (Weeks 7-8)
# ============================================================================

# Open-Meteo API settings (Phase 2)
WEATHER_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # All months
WEATHER_CACHE = CACHE_DIR / "weather"

# ============================================================================
# PHASE 3: Review Embeddings (Week 9)
# ============================================================================

# Sentence-BERT settings (Phase 3)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PHASE3_REVIEWS_TO_SAMPLE = 1000  # Sample for embedding to save compute
EMBEDDING_CACHE = CACHE_DIR / "embeddings"

# ============================================================================
# PHASE 4: Machine Learning Models (Week 10)
# ============================================================================

# Random Forest (Phase 4)
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_RANDOM_STATE = RANDOM_STATE

# Two-Tower Neural Network (Phase 4)
TOWER_EMBEDDING_DIM = 64
TOWER_HIDDEN_DIM = 128
TOWER_LEARNING_RATE = 0.001
TOWER_BATCH_SIZE = 32
TOWER_EPOCHS = 20

# ============================================================================
# PHASE 5: Ablation Experiments (Week 11)
# ============================================================================

# Feature combinations to test
FEATURE_CONFIGS = {
    "baseline": ["user_rating_behavior", "destination_stats"],
    "with_weather": ["user_rating_behavior", "destination_stats", "weather"],
    "with_budget": ["user_rating_behavior", "destination_stats", "price_level"],
    "with_embeddings": ["user_rating_behavior", "destination_stats", "review_embeddings"],
    "with_weather_budget": ["user_rating_behavior", "destination_stats", "weather", "price_level"],
    "full": ["user_rating_behavior", "destination_stats", "weather", "price_level", "review_embeddings"]
}