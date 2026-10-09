from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import time
from collections import defaultdict, deque
import calendar
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path so pickled models can find src.models
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Load API keys (UNSPLASH_API_KEY, ANTHROPIC_API_KEY) from backend/.env before
# the modules below read them at import time
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from backend.db import init_db, save_session, get_session, log_feedback, get_metrics
from backend.agent import chat_with_agent, explain_recommendations
from backend.scorer import score_cities
from backend.bandit import get_or_create_bandit
from backend.image_service import fetch_photo_sync, get_weather_condition

# Initialize FastAPI app
app = FastAPI(title="Travel Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Bandit context of each city last shown to a session, so feedback can update
# the bandit with the context the user actually reacted to.
_SHOWN_CONTEXTS: dict[str, dict[str, np.ndarray]] = {}
BANDIT_WEIGHT = 0.1
# Most cities are suburbs of ~11 Yelp metros (Philadelphia alone has 267), so
# each extra pick from the same metro costs this much score. Keeps the list
# varied without forcing in poor matches when only one metro truly fits.
METRO_REPEAT_PENALTY = 0.05

# Sections: a city must reach this fraction of the best final score to count
# as a good fit for "popular" / "hidden gem"; states need STATE_MIN_RATIO.
GOOD_MATCH_RATIO = 0.7
STATE_MIN_RATIO = 0.6
SECTION_SIZES = {"best": 5, "popular": 4, "gems": 4, "states": 6}

def diversify_by_metro(df, k, score_col="final_score", penalty=METRO_REPEAT_PENALTY):
    """Greedy top-k by score_col minus a penalty per earlier pick from the same metro."""
    remaining = df.copy()
    picked, per_metro = [], {}
    while len(picked) < k and len(remaining):
        adjusted = remaining[score_col] - penalty * remaining["metro"].map(per_metro).fillna(0)
        idx = adjusted.idxmax()
        picked.append(idx)
        metro = remaining.at[idx, "metro"]
        per_metro[metro] = per_metro.get(metro, 0) + 1
        remaining = remaining.drop(idx)
    # Show the chosen set best-first
    return df.loc[picked].sort_values(score_col, ascending=False, kind="stable")

def build_sections(df):
    """
    Split recommendations into themed rows. Each city appears in at most one
    row; rows are filled in order so "Best matches" gets first pick.
    Returns list of (key, title, description, DataFrame).
    """
    top = df["final_score"].max()
    good = df[df["final_score"] >= GOOD_MATCH_RATIO * top]
    small_town = df["popularity"] <= df["popularity"].median()
    shown = set()

    def take(candidates, k, score_col="final_score", penalty=METRO_REPEAT_PENALTY):
        candidates = candidates[~candidates.index.isin(shown)]
        picked = diversify_by_metro(candidates, k, score_col, penalty) if len(candidates) else candidates
        shown.update(picked.index)
        return picked

    # Light penalty here: the other rows already add variety, so this row stays truly "best"
    best = take(df, SECTION_SIZES["best"], penalty=0.02)
    # Popular: well-known places that still fit; strong metro penalty (popularity is 0-1)
    popular = take(good, SECTION_SIZES["popular"], "popularity", penalty=0.3)
    gems = take(good[small_town.loc[good.index]], SECTION_SIZES["gems"])

    remaining = df[~df.index.isin(shown)].sort_values("final_score", ascending=False)
    state_best = remaining.groupby("state").head(1)
    state_best = state_best[state_best["final_score"] >= STATE_MIN_RATIO * top]
    states = state_best.head(SECTION_SIZES["states"])

    return [
        ("best", "🎯 Best matches", "Highest overall fit to your preferences", best),
        ("popular", "🔥 Popular picks", "Well-known destinations that still fit you well", popular),
        ("gems", "💎 Hidden gems", "Smaller towns that match you well", gems),
        ("states", "🗺️ Best in each state", "Top remaining pick per state", states),
    ]

def reason_for(row, prefs):
    """Short data-based 'why' line for a card."""
    parts = [f"{row['dest_temp']:.0f}°C in {calendar.month_abbr[prefs.travel_month]}"]
    if prefs.activities and row["activity_sim"] >= 0.6:
        parts.append(f"strong {' & '.join(a.lower() for a in prefs.activities[:2])} scene")
    if row["budget_match"] >= 0.8:
        parts.append("fits your budget")
    parts.append(f"{int(row['num_businesses']):,} places on Yelp")
    return " · ".join(parts)

_PHOTO_CACHE: dict[tuple, dict] = {}

def cached_photo(city, weather_condition, travel_month):
    # Unsplash demo keys allow 50 requests/hour, so never fetch the same photo
    # twice. Only successes are cached: a miss (e.g. rate limited) retries later.
    key = (city, weather_condition)
    if key not in _PHOTO_CACHE:
        photo = fetch_photo_sync(city, weather_condition, travel_month)
        if not photo:
            return None
        _PHOTO_CACHE[key] = photo
    return _PHOTO_CACHE[key]

# Chat guardrails: per-session history (kept in memory) and usage limits
MAX_MESSAGE_CHARS = 500
MAX_USER_TURNS = 20          # per session
RATE_LIMIT_MESSAGES = 6      # per RATE_LIMIT_WINDOW_S seconds, per session
RATE_LIMIT_WINDOW_S = 60
_CHAT_HISTORY: dict[str, list[dict]] = defaultdict(list)
_CHAT_TIMES: dict[str, deque] = defaultdict(deque)

def chat_limit_reply(session_id: str, message: str) -> str | None:
    """Canned reply if this message breaks a chat limit, else None."""
    if not message:
        return "Tell me a bit about the trip you have in mind - for example, when you'd like to travel."
    if len(message) > MAX_MESSAGE_CHARS:
        return f"That message is a bit long - could you keep it under {MAX_MESSAGE_CHARS} characters?"
    if sum(m["role"] == "user" for m in _CHAT_HISTORY[session_id]) >= MAX_USER_TURNS:
        return "We've chatted a lot! Please use the form above to adjust your preferences."
    times = _CHAT_TIMES[session_id]
    now = time.monotonic()
    while times and now - times[0] > RATE_LIMIT_WINDOW_S:
        times.popleft()
    if len(times) >= RATE_LIMIT_MESSAGES:
        return "You're sending messages quickly - please wait a moment and try again."
    times.append(now)
    return None

# Pydantic models
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    agent_reply: str
    preferences_extracted: dict | None = None

class UserPreferences(BaseModel):
    travel_month: int = 6
    climate_preference: str = "moderate"
    activities: list = []
    budget: str = "medium"
    free_text_interests: str = ""

class RecommendRequest(BaseModel):
    session_id: str
    preferences: UserPreferences

class CityRecommendation(BaseModel):
    rank: int
    city: str
    country: str = "USA"
    state: str = ""
    metro: str = ""
    reason: str = ""  # short data-based "why", e.g. "16°C in Jan · fits your budget"
    rf_score: float
    climate_match: float
    activity_sim: float
    budget_match: float = 0.5
    popularity: float
    match_score: float = 0.0  # 0-1 overall fit to the submitted preferences
    explanation: str = ""
    image_url: str = ""  # Unsplash image URL
    image_credit: dict = {}  # {photographer, photographer_url, unsplash_url}
    weather: dict = {}  # Weather info (condition, temp, etc)

class RecommendationSection(BaseModel):
    key: str
    title: str
    description: str
    recommendations: list[CityRecommendation]

class RecommendResponse(BaseModel):
    session_id: str
    recommendations: list[CityRecommendation]  # all shown cities, flat (best matches first)
    sections: list[RecommendationSection] = []

class FeedbackRequest(BaseModel):
    session_id: str
    city: str
    reward: int  # 1 for thumbs up, 0 for thumbs down

@app.get("/health")
async def health_check():
    """Check API health and model loading."""
    try:
        from scorer import load_model
        load_model()
        return {"status": "healthy", "message": "Model loaded successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Chat with the conversational agent.
    Agent gathers user preferences through natural conversation.
    (Plain def: FastAPI runs it in a worker thread, so the blocking API call
    doesn't stall other requests.)
    """
    # Get or create session
    if not get_session(req.session_id):
        save_session(req.session_id, None)

    message = req.message.strip()
    limit_reply = chat_limit_reply(req.session_id, message)
    if limit_reply:
        return ChatResponse(session_id=req.session_id, agent_reply=limit_reply)

    history = _CHAT_HISTORY[req.session_id]
    agent_reply, extracted_prefs = chat_with_agent(req.session_id, message, list(history))
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": agent_reply}]

    # If preferences were extracted, save them
    if extracted_prefs:
        save_session(req.session_id, extracted_prefs)

    return ChatResponse(
        session_id=req.session_id,
        agent_reply=agent_reply,
        preferences_extracted=extracted_prefs
    )

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Get recommendations for the user based on their preferences.
    Uses Random Forest + Contextual Bandit scoring.
    """
    session_id = req.session_id
    prefs = req.preferences

    # Score all cities using the Random Forest
    scores_df = score_cities({
        "travel_month": prefs.travel_month,
        "climate_preference": prefs.climate_preference,
        "activities": prefs.activities,
        "budget": prefs.budget,
        "free_text_interests": prefs.free_text_interests
    })

    # Apply contextual bandit re-ranking
    bandit = get_or_create_bandit()

    # Build context matrix for bandit
    contexts = np.array([
        [row["rf_score"] / 5.0, row["climate_match"], row["activity_sim"], row["popularity"]]
        for _, row in scores_df.iterrows()
    ])

    # Bandit learns from thumbs up/down and nudges the preference-based ranking.
    # (Sorting by bandit score alone ignored the form: with no feedback yet its
    # scores are all zero.)
    bandit_scores = bandit.get_scores(contexts)
    scores_df["bandit_score"] = bandit_scores
    scores_df["final_score"] = scores_df["combined_score"] + BANDIT_WEIGHT * scores_df["bandit_score"]
    _SHOWN_CONTEXTS[session_id] = {row["city"]: ctx for (_, row), ctx in zip(scores_df.iterrows(), contexts)}

    scores_df = scores_df.sort_values("final_score", ascending=False, kind="stable").reset_index(drop=True)
    sections = build_sections(scores_df)
    shown_df = pd.concat([rows for *_, rows in sections])

    # LLM explanations only for the best matches (one API call per city)
    explanations = explain_recommendations(
        sections[0][3].to_dict('records'),
        {
            "travel_month": prefs.travel_month,
            "climate_preference": prefs.climate_preference,
            "activities": prefs.activities,
            "budget": prefs.budget
        }
    )

    # Travel-month weather is in deg C / mm; get_weather_condition expects deg F / inches
    conditions = {
        idx: get_weather_condition(row["dest_precip"] / 25.4, row["dest_temp"] * 9 / 5 + 32)
        for idx, row in shown_df.iterrows()
    }
    # Fetch weather-appropriate images from Unsplash in parallel
    with ThreadPoolExecutor(max_workers=8) as pool:
        photos = dict(zip(conditions, pool.map(
            lambda idx: cached_photo(shown_df.at[idx, "city"], conditions[idx], prefs.travel_month),
            conditions,
        )))

    def to_rec(idx, row, rank):
        return CityRecommendation(
            rank=rank,
            city=row["city"],
            country="Canada" if row["state"] == "AB" else "USA",
            state=str(row["state"]),
            metro=str(row["metro"]),
            rf_score=float(row["rf_score"]),
            climate_match=float(row["climate_match"]),
            activity_sim=float(row["activity_sim"]),
            budget_match=float(row["budget_match"]),
            popularity=float(row["popularity"]),
            match_score=float(row["combined_score"]),
            reason=reason_for(row, prefs),
            explanation=explanations.get(row["city"], ""),
            image_url=photos[idx]["url"] if photos[idx] else "",
            image_credit={k: v for k, v in (photos[idx] or {}).items() if k != "url"},
            weather={
                "condition": conditions[idx],
                "temp": float(row["dest_temp"] * 9 / 5 + 32),
                "rainfall": float(row["dest_precip"] / 25.4)
            }
        )

    response_sections = [
        RecommendationSection(
            key=key, title=title, description=description,
            recommendations=[to_rec(idx, row, i + 1) for i, (idx, row) in enumerate(rows.iterrows())],
        )
        for key, title, description, rows in sections
        if len(rows)
    ]
    recommendations = [rec for sec in response_sections for rec in sec.recommendations]

    # Save session with preferences
    save_session(session_id, prefs.dict())

    return RecommendResponse(
        session_id=session_id,
        recommendations=recommendations,
        sections=response_sections
    )

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Submit feedback (thumbs up/down) on a recommendation.
    Bandit learns from this feedback.
    """
    session_id = req.session_id
    city = req.city
    reward = req.reward  # 1 or 0

    # Log feedback to DB
    log_feedback(session_id, city, reward)

    # Update bandit with this feedback
    # (In a real system, we'd need to also store the context for this city)
    bandit = get_or_create_bandit()

    context = _SHOWN_CONTEXTS.get(session_id, {}).get(city)
    if context is not None:
        bandit.update(context, reward)

    return {"status": "success", "message": f"Feedback recorded: {city} -> {reward}"}

@app.get("/metrics")
async def get_system_metrics():
    """
    Get overall metrics: cumulative reward, feedback count, etc.
    """
    metrics = get_metrics()
    return {
        "total_feedback": metrics["total_feedback"],
        "total_reward": metrics["total_reward"],
        "avg_reward": metrics["avg_reward"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
