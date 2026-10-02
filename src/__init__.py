"""Context-Aware Travel Destination Recommendation using ML and Review Embeddings."""

__version__ = "0.1.0"
__author__ = "CS 582 Team"

from src.data.loader import load_businesses, load_reviews, load_users
from src.data.normalizer import normalize_city_names, get_major_cities
from src.features.user_features import UserProfileBuilder
from src.features.destination_features import DestinationProfileBuilder
from src.evaluation.harness import RecommendationHarness
from src.evaluation.metrics import recall_at_k, ndcg_at_k, mean_reciprocal_rank

__all__ = [
    'load_businesses',
    'load_reviews',
    'load_users',
    'normalize_city_names',
    'get_major_cities',
    'UserProfileBuilder',
    'DestinationProfileBuilder',
    'RecommendationHarness',
    'recall_at_k',
    'ndcg_at_k',
    'mean_reciprocal_rank'
]
