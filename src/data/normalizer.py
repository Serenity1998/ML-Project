"""Normalize city names and business categories."""
import pandas as pd
from difflib import SequenceMatcher


def normalize_city_names(df, city_column='city'):
    """Normalize city names (strip whitespace, title case)."""
    df = df.copy()
    df[city_column] = df[city_column].str.strip().str.title()
    return df


def get_major_cities(businesses_df, min_businesses=50, min_reviews=500):
    """Identify major cities with enough businesses and reviews."""
    city_stats = businesses_df.groupby('city').agg({
        'business_id': 'count',
        'review_count': 'sum'
    }).rename(columns={'business_id': 'num_businesses', 'review_count': 'total_reviews'})

    major_cities = city_stats[
        (city_stats['num_businesses'] >= min_businesses) &
        (city_stats['total_reviews'] >= min_reviews)
    ].index.tolist()

    return major_cities


def filter_to_major_cities(df, city_column, major_cities):
    """Keep only records from major cities."""
    return df[df[city_column].isin(major_cities)].copy()


def parse_categories(categories_str):
    """Parse category string into list."""
    if pd.isna(categories_str):
        return []
    if isinstance(categories_str, list):
        return categories_str
    return [c.strip() for c in str(categories_str).split(',')]


def extract_primary_category(categories_list):
    """Extract primary category from list."""
    if categories_list:
        return categories_list[0]
    return 'Unknown'
