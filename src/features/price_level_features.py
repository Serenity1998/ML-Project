"""Extract and predict price levels from tips. (New feature)"""
import pandas as pd
import numpy as np
import re


class PriceLevelFeatureBuilder:
    """Build price level features from Yelp data and tips text."""

    # Price keywords mapping
    CHEAP_KEYWORDS = [
        'cheap', 'affordable', 'budget', 'bargain', 'deal', 'inexpensive',
        'free', 'low price', 'low cost', 'discount', 'savings', 'economy'
    ]

    EXPENSIVE_KEYWORDS = [
        'expensive', 'pricey', 'costly', 'break the bank', 'splurge',
        'high price', 'premium', 'upscale', 'fine dining', 'steep'
    ]

    WORTH_KEYWORDS = [
        'worth', 'value', 'good value', 'fair price', 'reasonable',
        'worth the money', 'worth the price'
    ]

    def __init__(self):
        self.price_cache = {}

    def extract_price_keywords(self, text):
        """Extract price signals from text."""
        if not text or not isinstance(text, str):
            return {'cheap': 0, 'expensive': 0, 'worth': 0, 'price_mention': 0}

        text_lower = text.lower()

        # Count keyword matches
        cheap_count = sum(1 for kw in self.CHEAP_KEYWORDS if kw in text_lower)
        expensive_count = sum(1 for kw in self.EXPENSIVE_KEYWORDS if kw in text_lower)
        worth_count = sum(1 for kw in self.WORTH_KEYWORDS if kw in text_lower)

        # Detect price mentions (e.g., "$10", "$15.99")
        price_mention = len(re.findall(r'\$\d+', text_lower))

        return {
            'cheap': cheap_count,
            'expensive': expensive_count,
            'worth': worth_count,
            'price_mention': price_mention
        }

    def predict_price_from_tips(self, tips_df, business_id):
        """Predict price level from tips for a business.

        Returns:
            int: 1=cheap, 2=medium, 3=expensive, None=unknown
        """
        if business_id not in tips_df['business_id'].values:
            return None

        business_tips = tips_df[tips_df['business_id'] == business_id]

        if len(business_tips) == 0:
            return None

        # Aggregate signals from all tips
        all_signals = {
            'cheap': 0,
            'expensive': 0,
            'worth': 0,
            'price_mention': 0
        }

        for tip_text in business_tips['text'].dropna():
            signals = self.extract_price_keywords(tip_text)
            for key in all_signals:
                all_signals[key] += signals[key]

        # Simple classification
        total_tips = len(business_tips)

        # If strong cheap signals
        if all_signals['cheap'] > all_signals['expensive']:
            return 1  # Budget

        # If strong expensive signals
        elif all_signals['expensive'] > all_signals['cheap']:
            return 3  # Expensive

        # Default to medium if any price signals mentioned
        elif sum(all_signals.values()) > 0:
            return 2  # Medium

        return None

    def build_price_features(self, businesses_df, tips_df):
        """Build price level features for all businesses.

        Strategy:
        1. Use existing Yelp RestaurantsPriceRange2 if available
        2. Predict from tips if missing
        3. Default to medium if no data

        Args:
            businesses_df: DataFrame with business data
            tips_df: DataFrame with tips data

        Returns:
            DataFrame with business_id and price_level (1-3 scale)
        """
        price_levels = []

        for idx, biz in businesses_df.iterrows():
            biz_id = biz['business_id']

            # Try to get Yelp's price range first
            yelp_price = None
            try:
                if 'attributes' in biz.index and isinstance(biz['attributes'], dict):
                    yelp_price = biz['attributes'].get('RestaurantsPriceRange2')
                    if yelp_price and yelp_price != 'None':
                        yelp_price = int(yelp_price)
                    else:
                        yelp_price = None
            except (KeyError, TypeError, ValueError):
                pass

            # If Yelp has price, use it
            if yelp_price and yelp_price in [1, 2, 3, 4]:
                # Map Yelp's 1-4 scale to our 1-3 scale
                # 1 -> 1, 2 -> 1, 3 -> 2, 4 -> 3
                if yelp_price <= 2:
                    predicted_price = 1
                elif yelp_price == 3:
                    predicted_price = 2
                else:
                    predicted_price = 3
                source = 'yelp'
            else:
                # Predict from tips
                predicted_price = self.predict_price_from_tips(tips_df, biz_id)
                source = 'tips'

                # Default to medium if no tips
                if predicted_price is None:
                    predicted_price = 2
                    source = 'default'

            price_levels.append({
                'business_id': biz_id,
                'business_name': biz.get('name', ''),
                'price_level': predicted_price,
                'price_source': source  # For debugging
            })

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(businesses_df)} businesses")

        return pd.DataFrame(price_levels)

    def get_price_label(self, price_level):
        """Convert numeric price level to human-readable label."""
        labels = {1: 'Low', 2: 'Medium', 3: 'High'}
        return labels.get(price_level, 'Unknown')
