"""
Service to fetch weather-appropriate images from Unsplash API
"""
import aiohttp
import os
from typing import Optional

UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY", "")  # free key: unsplash.com/developers
UNSPLASH_API_BASE = "https://api.unsplash.com/search/photos"

# Weather condition to search keywords mapping
WEATHER_KEYWORDS = {
    "EXTREME_HOT": "desert hot weather landscape",
    "HOT_SUNNY": "beach sunny weather palm trees",
    "PLEASANT": "park outdoor activities pleasant weather",
    "PARTLY_RAINY": "autumn rain cozy weather",
    "RAINY": "thunderstorm rain weather",
    "COLD": "winter cold snow weather",
    "SNOWY": "snow mountains winter landscape"
}

def get_weather_condition(rainfall: float, temp: float, humidity: float = 50) -> str:
    """Classify weather condition based on metrics"""
    if temp > 95:
        return "EXTREME_HOT"
    elif temp > 80 and rainfall < 1:
        return "HOT_SUNNY"
    elif rainfall > 3:
        return "RAINY"
    elif temp < 35:
        return "SNOWY"
    elif temp < 50:
        return "COLD"
    elif rainfall > 2:
        return "PARTLY_RAINY"
    else:
        return "PLEASANT"

async def fetch_image_from_unsplash(
    city: str,
    weather_condition: str,
    travel_month: int = 6
) -> Optional[str]:
    """
    Fetch an image URL from Unsplash based on city and weather condition.

    Args:
        city: City name (e.g., "Tampa")
        weather_condition: Weather condition (e.g., "RAINY")
        travel_month: Month (1-12) for additional context

    Returns:
        Image URL from Unsplash or None if request fails
    """
    try:
        # Construct search query: city + weather keywords
        weather_keywords = WEATHER_KEYWORDS.get(weather_condition, "travel destination")
        query = f"{city} {weather_keywords}"

        params = {
            "query": query,
            "per_page": 1,
            "order_by": "relevant",
            "client_id": UNSPLASH_API_KEY
        }

        # Make async request to Unsplash
        async with aiohttp.ClientSession() as session:
            async with session.get(UNSPLASH_API_BASE, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("results") and len(data["results"]) > 0:
                        image_url = data["results"][0]["urls"]["regular"]
                        return image_url

        # Fallback: Return generic city image if specific weather image fails
        fallback_params = {
            "query": city,
            "per_page": 1,
            "order_by": "relevant",
            "client_id": UNSPLASH_API_KEY
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(UNSPLASH_API_BASE, params=fallback_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("results") and len(data["results"]) > 0:
                        return data["results"][0]["urls"]["regular"]

        return None

    except Exception as e:
        print(f"Error fetching image for {city}: {e}")
        return None

async def fetch_images_for_recommendations(
    recommendations: list,
    travel_month: int,
    weather_data: dict
) -> list:
    """
    Fetch images for all recommendations.

    Args:
        recommendations: List of recommendation dicts with 'city' key
        travel_month: Travel month for context
        weather_data: Dict with city → weather_condition mapping

    Returns:
        List of recommendations with added 'image_url' field
    """
    for rec in recommendations:
        city = rec["city"]

        # Get weather condition for this city
        if city in weather_data:
            weather_condition = weather_data[city]["condition"]
        else:
            # Default to PLEASANT if no weather data
            weather_condition = "PLEASANT"

        # Fetch image asynchronously
        image_url = await fetch_image_from_unsplash(city, weather_condition, travel_month)
        rec["image_url"] = image_url or ""

        # Add weather info for display
        if city in weather_data:
            rec["weather"] = weather_data[city]

    return recommendations

# Unsplash API guidelines: credit the photographer and link back to Unsplash
# with these referral parameters.
UNSPLASH_APP_NAME = os.getenv("UNSPLASH_APP_NAME", "travel_recommender")
UTM = f"utm_source={UNSPLASH_APP_NAME}&utm_medium=referral"

def _search_photo(query: str) -> Optional[dict]:
    """First Unsplash search result as {url, photographer, photographer_url, unsplash_url}."""
    import requests
    params = {"query": query, "per_page": 1, "order_by": "relevant", "client_id": UNSPLASH_API_KEY}
    response = requests.get(UNSPLASH_API_BASE, params=params, timeout=5)
    if response.status_code != 200:
        return None
    results = response.json().get("results")
    if not results:
        return None
    photo = results[0]
    return {
        "url": photo["urls"]["regular"],
        "photographer": photo["user"]["name"],
        "photographer_url": f"{photo['user']['links']['html']}?{UTM}",
        "unsplash_url": f"https://unsplash.com/?{UTM}",
    }

# Synchronous versions for FastAPI
def fetch_photo_sync(city: str, weather_condition: str, travel_month: int = 6) -> Optional[dict]:
    """Weather-themed photo of the city, falling back to any photo of the city."""
    if not UNSPLASH_API_KEY:
        return None
    try:
        weather_keywords = WEATHER_KEYWORDS.get(weather_condition, "travel destination")
        return _search_photo(f"{city} {weather_keywords}") or _search_photo(city)
    except Exception as e:
        print(f"Error fetching image for {city}: {e}")
        return None

def fetch_image_sync(city: str, weather_condition: str, travel_month: int = 6) -> Optional[str]:
    """Image URL only (see fetch_photo_sync for photographer credit)."""
    photo = fetch_photo_sync(city, weather_condition, travel_month)
    return photo["url"] if photo else None
