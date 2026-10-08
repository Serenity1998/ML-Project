from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import numpy as np
import sys
from pathlib import Path

# Add project root to path so pickled models can find src.models
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from backend.db import init_db, save_session, get_session, log_feedback, get_metrics
from backend.agent import chat_with_agent, explain_recommendations
from backend.scorer import score_cities
from backend.bandit import get_or_create_bandit

# Initialize FastAPI app
app = FastAPI(title="Travel Recommender API")

# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Pydantic models
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    agent_reply: str
    preferences_extracted: dict = None

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
    rf_score: float
    climate_match: float
    activity_sim: float
    popularity: float
    explanation: str = ""

class RecommendResponse(BaseModel):
    session_id: str
    recommendations: list[CityRecommendation]

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
async def chat(req: ChatRequest):
    """
    Chat with the conversational agent.
    Agent gathers user preferences through natural conversation.
    """
    # Get or create session
    session = get_session(req.session_id)
    if not session:
        session = {"session_id": req.session_id, "user_prefs": None, "message_count": 0}
        save_session(req.session_id, None)

    # Build conversation history (simplified: just use the message count)
    # In production, store full history in DB
    history = []  # Agent doesn't need full history for fallback

    # Chat with agent
    agent_reply, extracted_prefs = chat_with_agent(
        req.session_id,
        req.message,
        history
    )

    # If preferences were extracted, save them
    if extracted_prefs:
        save_session(req.session_id, extracted_prefs)

    return ChatResponse(
        session_id=req.session_id,
        agent_reply=agent_reply,
        preferences_extracted=extracted_prefs
    )

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
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

    # Get bandit scores
    bandit_scores = bandit.get_scores(contexts)
    scores_df["bandit_score"] = bandit_scores

    # Rerank by bandit
    scores_df = scores_df.sort_values("bandit_score", ascending=False).reset_index(drop=True)

    # Get explanations (optional, only for top 5)
    explanations = explain_recommendations(
        scores_df.head(5).to_dict('records'),
        {
            "travel_month": prefs.travel_month,
            "climate_preference": prefs.climate_preference,
            "activities": prefs.activities,
            "budget": prefs.budget
        }
    )

    # Format response
    recommendations = [
        CityRecommendation(
            rank=i + 1,
            city=row["city"],
            rf_score=float(row["rf_score"]),
            climate_match=float(row["climate_match"]),
            activity_sim=float(row["activity_sim"]),
            popularity=float(row["popularity"]),
            explanation=explanations.get(row["city"], "")
        )
        for i, (_, row) in enumerate(scores_df.head(10).iterrows())
    ]

    # Save session with preferences
    save_session(session_id, prefs.dict())

    return RecommendResponse(
        session_id=session_id,
        recommendations=recommendations
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

    # Dummy context for now (in production, store context with each recommendation)
    context = np.array([0.5, 0.5, 0.5, 0.5])
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
