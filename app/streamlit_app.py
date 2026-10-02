"""Streamlit app for cross-city Yelp recommender."""
import sys
sys.path.insert(0, '../')

import streamlit as st
import pandas as pd
from config import *
from src.data_loader import load_businesses, load_reviews
from src.recommender import BaselineRecommender, ContentBasedRecommender


st.set_page_config(page_title="Yelp Cross-City Recommender", layout="wide")

st.title("🍽️ Cross-City Yelp Recommender")

st.markdown("""
Discover great restaurants and businesses across cities based on your review history.
""")

# Load data
@st.cache_resource
def load_data():
    businesses = load_businesses(BUSINESS_FILE)
    reviews = pd.read_parquet(PROCESSED_DATA_DIR / 'reviews_filtered.parquet')
    return businesses, reviews


try:
    businesses, reviews = load_data()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("User Settings")
        user_id = st.selectbox(
            "Select a user:",
            reviews['user_id'].unique()
        )

    with col2:
        st.subheader("Recommendation Settings")
        target_city = st.selectbox(
            "Target city:",
            businesses['city'].unique()
        )
        n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)

    # Choose recommender
    recommender_choice = st.radio(
        "Choose recommender:",
        ["Baseline (Top-Rated)", "Content-Based"]
    )

    if st.button("Get Recommendations"):
        if recommender_choice == "Baseline (Top-Rated)":
            recommender = BaselineRecommender(businesses, reviews)
        else:
            recommender = ContentBasedRecommender(businesses, reviews)
            recommender.fit()

        recommendations = recommender.recommend(user_id, target_city, n_recommendations)

        if recommendations is not None and len(recommendations) > 0:
            st.subheader(f"Top {len(recommendations)} recommendations in {target_city}")
            st.dataframe(recommendations, use_container_width=True)
        else:
            st.warning("No recommendations available for this user-city combination.")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Make sure the data files are in the `data/raw/` directory.")
