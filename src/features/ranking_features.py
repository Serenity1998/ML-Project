"""Features for ranking (user, city) pairs."""
import numpy as np
import pandas as pd

USER_FEATURES = ['user_avg_stars', 'user_std_stars', 'user_log_reviews', 'user_num_cities', 'home_lat', 'home_lon']
CITY_FEATURES = ['city_log_businesses', 'city_avg_stars', 'city_log_reviews', 'lat', 'lon']
PAIR_FEATURES = ['distance_km']
WEATHER_FEATURES = ['city_temp', 'city_precip', 'temp_diff']


def city_features(businesses_df):
    """Per-city stats and center coordinates from Yelp businesses."""
    grouped = businesses_df.groupby('city')
    return pd.DataFrame({
        'city_log_businesses': np.log1p(grouped.size()),
        'city_avg_stars': grouped['stars'].mean(),
        'city_log_reviews': np.log1p(grouped['review_count'].sum()),
        'lat': grouped['latitude'].median(),
        'lon': grouped['longitude'].median(),
    })


def user_features(history_reviews, businesses_df, cities_df):
    """Per-user rating behavior from history only, plus home city (most-reviewed) coordinates."""
    merged = history_reviews.merge(businesses_df[['business_id', 'city']], on='business_id')
    grouped = merged.groupby('user_id')
    users = pd.DataFrame({
        'user_avg_stars': grouped['stars'].mean(),
        'user_std_stars': grouped['stars'].std().fillna(0.0),
        'user_log_reviews': np.log1p(grouped.size()),
        'user_num_cities': grouped['city'].nunique(),
        'home_city': grouped['city'].agg(lambda s: s.value_counts().index[0]),
    })
    home = cities_df[['lat', 'lon']].rename(columns={'lat': 'home_lat', 'lon': 'home_lon'})
    return users.join(home, on='home_city')


def trip_months(reviews_df, businesses_df, targets_df):
    """Month of each user's first review in their held-out city: the trip's travel month."""
    merged = reviews_df.merge(businesses_df[['business_id', 'city']], on='business_id')
    merged = merged.merge(targets_df, on=['user_id', 'city'])
    return pd.to_datetime(merged['date']).groupby(merged['user_id']).min().dt.month


def city_month_weather(cities_df, weather_builder):
    """Monthly temperature and precipitation per city, fetched at Yelp business coordinates."""
    rows = []
    for city, row in cities_df.iterrows():
        monthly = weather_builder.compute_monthly_features(
            weather_builder.fetch_historical_weather(row['lat'], row['lon'])
        )
        if monthly is not None:
            rows.append(monthly[['month', 'avg_temp', 'total_precipitation']].assign(city=city))
    weather_builder._save_caches()
    return pd.concat(rows, ignore_index=True).set_index(['city', 'month'])


def build_pair_features(pairs_df, users_df, cities_df, months, weather=None):
    """Join user, city and pair features onto (user_id, city) rows; add trip-month weather if given."""
    df = pairs_df.join(users_df, on='user_id').join(cities_df, on='city')

    lat1, lon1, lat2, lon2 = (np.radians(df[c]) for c in ['home_lat', 'home_lon', 'lat', 'lon'])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df['distance_km'] = 6371 * 2 * np.arcsin(np.sqrt(a))

    if weather is not None:
        month = df['user_id'].map(months)
        city_key = pd.MultiIndex.from_arrays([df['city'], month])
        home_key = pd.MultiIndex.from_arrays([df['home_city'], month])
        df['city_temp'] = weather['avg_temp'].reindex(city_key).to_numpy()
        df['city_precip'] = weather['total_precipitation'].reindex(city_key).to_numpy()
        df['temp_diff'] = df['city_temp'] - weather['avg_temp'].reindex(home_key).to_numpy()
        df[WEATHER_FEATURES] = df[WEATHER_FEATURES].fillna(df[WEATHER_FEATURES].median())

    return df
