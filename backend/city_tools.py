"""
Grounded city facts for the chat agent's tools.

Everything here comes from the city table (Yelp business data + Open-Meteo
climate), so the agent can answer questions about a city without inventing
facts.
"""
import calendar
import difflib
from typing import Optional

import numpy as np

try:
    from backend.city_table import get_city_table, get_city_profiles
    from backend.scorer import ACTIVITY_CATEGORIES
except ImportError:
    from city_table import get_city_table, get_city_profiles
    from scorer import ACTIVITY_CATEGORIES

# Categories worth mentioning to a traveler (skips dentists, auto repair, ...)
TRAVEL_CATEGORIES = sorted({c for cats in ACTIVITY_CATEGORIES.values() for c in cats} | {
    "Wine Bars", "Breweries", "Coffee & Tea", "Bakeries", "Music Venues", "Golf",
    "Hotels", "Tours", "Art Galleries", "Zoos", "Aquariums", "Botanical Gardens",
})
PRICE_LABELS = {1: "$ (inexpensive)", 2: "$$ (moderate)", 3: "$$$ (pricey)", 4: "$$$$ (very pricey)"}


def resolve_city(name: str) -> Optional[str]:
    """Map user/model spelling ("tampa", "St. Pete Beach") to a city key like "Tampa, FL"."""
    table = get_city_table()
    if name in table.index:
        return name
    lower = {k.lower(): k for k in table.index}
    if name.lower() in lower:
        return lower[name.lower()]
    # Name without state: pick the city with that name that has most businesses
    bare = name.split(",")[0].strip().lower().replace("st. ", "saint ").replace("st ", "saint ")
    same_name = table[table["city_name"].str.lower() == bare] if "city_name" in table else table.iloc[0:0]
    if len(same_name):
        return same_name["dest_num_businesses"].idxmax()
    close = difflib.get_close_matches(name.lower(), list(lower), n=1, cutoff=0.75)
    return lower[close[0]] if close else None


_HOME_CACHE: dict[str, dict] = {}


def _home_monthly_temps(lat: float, lon: float) -> Optional[dict]:
    """Average temperature (deg C) per month at a point, from Open-Meteo's archive (cached on disk)."""
    try:
        from config import WEATHER_CACHE
        from src.features.weather_features import WeatherFeatureBuilder
        builder = WeatherFeatureBuilder(cache_dir=WEATHER_CACHE)
        monthly = builder.compute_monthly_features(builder.fetch_historical_weather(lat, lon))
        builder._save_caches()
    except Exception:
        return None
    if monthly is None or len(monthly) < 12:
        return None
    return {int(m): float(t) for m, t in zip(monthly["month"], monthly["avg_temp"])}


def resolve_home(name: str) -> Optional[dict]:
    """
    Where the user is traveling from: {"label", "lat", "lon", "temps"} with temps = month -> deg C (or None).
    Catalog cities are matched exactly (no fuzzy matching: a wrong home would silently skew results);
    anything else is geocoded with Open-Meteo. Returns None if the place can't be found.
    """
    name = name.strip()
    if not name:
        return None
    if name.lower() in _HOME_CACHE:
        return _HOME_CACHE[name.lower()]

    table = get_city_table()
    lower = {k.lower(): k for k in table.index}
    bare = name.split(",")[0].strip().lower()
    key = lower.get(name.lower())
    if key is None and "," not in name:
        same_name = table[table["city_name"].str.lower() == bare]
        key = same_name["dest_num_businesses"].idxmax() if len(same_name) else None

    if key is not None:
        row = table.loc[key]
        home = {"label": key, "lat": float(row["dest_lat"]), "lon": float(row["dest_lon"]),
                "temps": {m: float(row[f"dest_temp_month_{m}"]) for m in range(1, 13)}}
    else:
        try:
            import requests
            response = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                                    params={"name": name.split(",")[0].strip(), "count": 1}, timeout=5)
            hit = (response.json().get("results") or [None])[0]
        except Exception:
            return None  # network trouble: not cached, so the next request retries
        if hit is None:
            return None
        label = ", ".join(p for p in [hit.get("name"), hit.get("admin1"), hit.get("country_code")] if p)
        home = {"label": label, "lat": float(hit["latitude"]), "lon": float(hit["longitude"]),
                "temps": _home_monthly_temps(hit["latitude"], hit["longitude"])}

    _HOME_CACHE[name.lower()] = home
    return home


def suggest_cities(name: str, n: int = 3) -> list[str]:
    table = get_city_table()
    return difflib.get_close_matches(name, list(table.index), n=n, cutoff=0.5)


def city_details(name: str, travel_month: Optional[int] = None) -> dict:
    """Facts about one city, or {"error": ...} if it isn't in the catalog."""
    key = resolve_city(name)
    if key is None:
        return {"error": f"'{name}' is not in this app's city catalog.", "did_you_mean": suggest_cities(name)}

    table = get_city_table()
    category_share, avg_price = get_city_profiles()
    row = table.loc[key]

    # Activity strengths: percentile of this city among all cities, per activity
    strengths = {}
    for activity, cats in ACTIVITY_CATEGORIES.items():
        cats = [c for c in cats if c in category_share.columns]
        if cats:
            scores = category_share[cats].sum(axis=1)
            strengths[activity] = int(round((scores < scores[key]).mean() * 100))
    top = sorted(strengths.items(), key=lambda kv: -kv[1])

    # Categories this city has unusually many of, compared with other cities
    cats = [c for c in TRAVEL_CATEGORIES if c in category_share.columns]
    lift = category_share.loc[key, cats] / category_share[cats].mean()
    n_businesses = int(row.get("dest_num_businesses", 0))
    enough = category_share.loc[key, cats] * n_businesses >= 3
    notable = lift[enough].sort_values(ascending=False).head(6).index.tolist()

    price = float(avg_price[key])
    details = {
        "city": key,
        "metro_area": row.get("metro", ""),
        "places_on_yelp": n_businesses,
        "price_level": PRICE_LABELS[int(np.clip(round(price), 1, 4))],
        "cheaper_than_pct_of_cities": int(round((avg_price > price).mean() * 100)),
        "avg_temp_c_by_month": {
            calendar.month_abbr[m]: round(float(row[f"dest_temp_month_{m}"]), 1) for m in range(1, 13)
        },
        "strongest_activities_percentile": dict(top[:3]),
        "weakest_activities_percentile": dict(top[-2:]),
        "notable_place_types": notable,
    }
    if travel_month:
        details["travel_month"] = {
            "month": calendar.month_name[travel_month],
            "avg_temp_c": round(float(row[f"dest_temp_month_{travel_month}"]), 1),
            "rain_mm": round(float(row[f"dest_precip_month_{travel_month}"])),
        }
    return details


def available_states() -> list[str]:
    return sorted(get_city_table()["state"].unique())
