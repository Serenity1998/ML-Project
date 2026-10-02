"""Review + Tips embedding utilities for Phase 3."""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


class ReviewTipsEmbedder:
    """Generate embeddings for review text + tips. (Phase 3)"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize Sentence-BERT model. (Phase 3)"""
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}

    def combine_review_and_tips(self, reviews_df, tips_df):
        """Combine review text with tips for each business. (Phase 3)"""
        combined_texts = []

        for business_id in reviews_df['business_id'].unique():
            biz_reviews = reviews_df[reviews_df['business_id'] == business_id]
            biz_tips = tips_df[tips_df['business_id'] == business_id]

            # Collect all text
            texts = []
            if 'text' in biz_reviews.columns:
                texts.extend(biz_reviews['text'].dropna().tolist())
            if 'text' in biz_tips.columns:
                texts.extend(biz_tips['text'].dropna().tolist())

            # Combine into single text
            combined = ' '.join(texts[:100])  # Limit to prevent huge texts
            combined_texts.append({
                'business_id': business_id,
                'combined_text': combined
            })

        return pd.DataFrame(combined_texts)

    def embed_texts(self, texts):
        """Generate embeddings for list of texts. (Phase 3)"""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)

    def aggregate_user_embeddings(self, user_id, reviews_df, tips_df, businesses_df):
        """Aggregate review+tip embeddings to user profile. (Phase 3)"""
        # Get user's reviews
        user_reviews = reviews_df[reviews_df['user_id'] == user_id]
        merged = user_reviews.merge(
            businesses_df[['business_id']],
            on='business_id'
        )

        # Get corresponding tips
        user_tips = tips_df[tips_df['user_id'] == user_id]

        # Combine texts
        all_texts = []
        if 'text' in user_reviews.columns:
            all_texts.extend(user_reviews['text'].dropna().tolist())
        if 'text' in user_tips.columns:
            all_texts.extend(user_tips['text'].dropna().tolist())

        if not all_texts:
            return None

        # Generate embeddings
        embeddings = self.embed_texts(all_texts)

        # Average embedding
        return np.mean(embeddings, axis=0)

    def aggregate_destination_embeddings(self, city, reviews_df, tips_df, businesses_df):
        """Aggregate review+tip embeddings to destination profile. (Phase 3)"""
        # Get businesses in city
        city_businesses = businesses_df[businesses_df['city'] == city]['business_id'].tolist()

        # Get reviews for businesses in city
        city_reviews = reviews_df[reviews_df['business_id'].isin(city_businesses)]
        city_tips = tips_df[tips_df['business_id'].isin(city_businesses)]

        # Combine texts
        all_texts = []
        if 'text' in city_reviews.columns:
            all_texts.extend(city_reviews['text'].dropna().tolist())
        if 'text' in city_tips.columns:
            all_texts.extend(city_tips['text'].dropna().tolist())

        if not all_texts:
            return None

        # Sample if too many
        if len(all_texts) > 1000:
            all_texts = all_texts[:1000]

        # Generate embeddings
        embeddings = self.embed_texts(all_texts)

        # Average embedding
        return np.mean(embeddings, axis=0)
