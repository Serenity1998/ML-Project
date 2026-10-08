"""Fetch and compute weather features. (Phase 2)"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
import json
import time


class WeatherFeatureBuilder:
    """Build weather-based features from Open-Meteo data. (Phase 2)"""

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/weather")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.geocoding_cache_file = self.cache_dir / "geocoding_cache.json"
        self.weather_cache_file = self.cache_dir / "weather_cache.json"
        self._load_caches()

    def _load_caches(self):
        """Load existing caches to avoid redundant API calls."""
        self.geocoding_cache = {}
        self.weather_cache = {}

        if self.geocoding_cache_file.exists():
            with open(self.geocoding_cache_file) as f:
                self.geocoding_cache = json.load(f)

        if self.weather_cache_file.exists():
            with open(self.weather_cache_file) as f:
                self.weather_cache = json.load(f)

    def _save_caches(self):
        """Save caches to disk."""
        with open(self.geocoding_cache_file, 'w') as f:
            json.dump(self.geocoding_cache, f, indent=2)
        with open(self.weather_cache_file, 'w') as f:
            json.dump(self.weather_cache, f, indent=2)

    def get_city_coordinates(self, city_name):
        """Get latitude/longitude for city using Open-Meteo Geocoding API."""
        if city_name in self.geocoding_cache:
            return self.geocoding_cache[city_name]

        try:
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()
            if "results" not in data or len(data["results"]) == 0:
                return None

            result = data["results"][0]
            coords = {
                "lat": result.get("latitude"),
                "lon": result.get("longitude"),
                "name": result.get("name"),
                "country": result.get("country")
            }
            self.geocoding_cache[city_name] = coords
            return coords
        except Exception as e:
            print(f"Error fetching coordinates for {city_name}: {e}")
            return None

    def fetch_historical_weather(self, lat, lon, year=2023):
        """Fetch historical weather from Open-Meteo Archive API for all months."""
        cache_key = f"{lat:.2f}_{lon:.2f}_{year}"
        if cache_key in self.weather_cache:
            return self.weather_cache[cache_key]

        try:
            url = "https://archive-api.open-meteo.com/v1/archive"
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto"
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            self.weather_cache[cache_key] = data
            time.sleep(0.5)  # Rate limiting
            return data
        except Exception as e:
            print(f"Error fetching weather for ({lat}, {lon}): {e}")
            return None

    def compute_monthly_features(self, weather_data):
        """Extract monthly temperature and rainfall statistics from daily weather data."""
        if not weather_data or "daily" not in weather_data:
            return None

        daily = weather_data["daily"]
        dates = pd.to_datetime(daily["time"])
        temps_max = np.array(daily["temperature_2m_max"])
        temps_min = np.array(daily["temperature_2m_min"])
        precip = np.array(daily["precipitation_sum"])

        monthly_features = []
        for month in range(1, 13):
            mask = dates.month == month
            if not mask.any():
                continue

            month_data = {
                "month": month,
                "avg_temp_max": np.mean(temps_max[mask]),
                "avg_temp_min": np.mean(temps_min[mask]),
                "avg_temp": (np.mean(temps_max[mask]) + np.mean(temps_min[mask])) / 2,
                "total_precipitation": np.sum(precip[mask]),
                "avg_precipitation": np.mean(precip[mask]),
                "temp_std": np.std((temps_max[mask] + temps_min[mask]) / 2)
            }
            monthly_features.append(month_data)

        return pd.DataFrame(monthly_features)

    def build_weather_features(self, cities_df):
        """Build weather feature matrix for all cities.

        Args:
            cities_df: DataFrame with at least 'city' column

        Returns:
            DataFrame with city + weather features
        """
        all_weather = []

        for idx, row in cities_df.iterrows():
            city = row["city"]
            print(f"Processing {idx + 1}/{len(cities_df)}: {city}")

            # Get coordinates
            coords = self.get_city_coordinates(city)
            if not coords:
                print(f"  ⚠ Could not find coordinates for {city}")
                continue

            # Fetch weather
            weather_data = self.fetch_historical_weather(coords["lat"], coords["lon"])
            if not weather_data:
                print(f"  ⚠ Could not fetch weather for {city}")
                continue

            # Compute monthly features
            monthly_df = self.compute_monthly_features(weather_data)
            if monthly_df is None:
                print(f"  ⚠ Could not compute features for {city}")
                continue

            monthly_df["city"] = city
            monthly_df["lat"] = coords["lat"]
            monthly_df["lon"] = coords["lon"]
            all_weather.append(monthly_df)
            print(f"  ✓ Got {len(monthly_df)} months of data")

        if not all_weather:
            return pd.DataFrame()

        self._save_caches()
        return pd.concat(all_weather, ignore_index=True)
