"""Fetch and compute weather features. (Phase 2)"""
import pandas as pd
import numpy as np


class WeatherFeatureBuilder:
    """Build weather-based features from Open-Meteo data. (Phase 2)"""

    def __init__(self):
        pass

    def get_city_coordinates(self, city_name):
        """Get latitude/longitude for city. (Phase 2)"""
        pass

    def fetch_historical_weather(self, lat, lon, month):
        """Fetch historical weather from Open-Meteo. (Phase 2)"""
        pass

    def compute_monthly_features(self, city, month):
        """Compute temperature and rainfall for specific month. (Phase 2)"""
        pass

    def build_weather_features(self, cities, months):
        """Build weather feature matrix. (Phase 2)"""
        pass
