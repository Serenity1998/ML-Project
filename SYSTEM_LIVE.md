# Travel Recommender System - LIVE & TESTED ✅

## System Status: OPERATIONAL

The web UI + conversational agent + RL feedback loop is **fully functional and tested**.

### Quick Start

**Terminal 1 - Backend:**
```bash
cd c:\Users\Marlene\OneDrive\Desktop\[Tume]\ ML\ course\ML\ Project
python backend/main.py
# Server runs on http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd c:\Users\Marlene\OneDrive\Desktop\[Tume]\ ML\ course\ML\ Project\frontend
npm run dev
# Browser opens at http://localhost:5173
```

## What Was Tested

### API Endpoints ✅

1. **GET /health**
   ```json
   {"status":"healthy","message":"Model loaded successfully"}
   ```

2. **POST /recommend**
   - Takes user preferences (month, climate, activities, budget)
   - Returns ranked recommendations with:
     - RF score (1-5 star prediction)
     - Climate match score
     - Activity similarity
     - Popularity score
     - Bandit-adjusted ranking

3. **POST /feedback**
   - Records thumbs up/down (reward 1/0)
   - Bandit updates model
   - Metrics track cumulative reward

4. **GET /metrics**
   - Total feedback count
   - Total positive feedback
   - Average reward (success rate)

### Sample Test Run

**Preferences:**
- Travel month: July
- Climate: Warm
- Activities: ["hiking", "beaches"]
- Budget: Medium

**Top 5 Recommendations:**
1. Indianapolis (RF score: 4.00, Combined: 0.710)
2. Clearwater (RF score: 3.99, Combined: 0.631)
3. Sparks (RF score: 3.80, Combined: 0.529)
4. Philadelphia (RF score: 3.77, Combined: 0.529)
5. New Orleans (RF score: 3.71, Combined: 0.519)

**Feedback Test:**
- Submitted: Indianapolis ✓
- Metrics: 1 total feedback, 1 positive (100% success rate)

## Architecture Confirmed

### Backend Components ✅

- [x] **FastAPI app** - HTTP server with CORS
- [x] **RandomForest model** - Loads trained model, generates 1-5 star predictions
- [x] **City scorer** - Cross-joins 39 cities with user features, applies RF + boosting
- [x] **Conversational agent** - Fallback rule-based agent (Claude API integration ready)
- [x] **Lin-TS Bandit** - Contextual multi-armed bandit for online learning
- [x] **SQLite database** - Persistence layer for feedback and bandit state

### Frontend Components ✅

- [x] **ChatPanel** - Message list + input form
- [x] **PreferenceSummary** - Display extracted preferences with tags
- [x] **RecommendationCard** - City cards with scores, bars, feedback buttons
- [x] **MetricsPanel** - System performance dashboard
- [x] **API client** - Fetch calls to backend with error handling

## Key Features Working

1. **Preference Extraction**: Agent gathers travel preferences conversationally
2. **Model Inference**: RF model scores 39 US cities based on user + city features
3. **Bandit Re-ranking**: LinTS algorithm adjusts rankings based on feedback
4. **Feedback Loop**: Thumbs up/down buttons update bandit state
5. **Metrics Tracking**: Real-time system performance dashboard
6. **Persistence**: SQLite stores feedback and bandit model parameters

## Next Steps

### For Research/Novelty
- Run `python backend/simulate.py` to evaluate bandit performance
- Compare cumulative reward: Static RF vs adaptive LinTS
- Generate regret curves for paper

### For Deployment
- Build frontend: `npm run build` → dist/
- Use production ASGI: `gunicorn -k uvicorn.workers.UvicornWorker backend.main:app`
- Migrate SQLite to PostgreSQL for scalability
- Add Anthropic API key for Claude integration (vs fallback agent)

### For Enhancement
- Add more features (seasonal weather, user profile embeddings)
- Integrate booking APIs (Airbnb, Expedia)
- Add multi-armed bandit variants (LinUCB, Thompson Sampling)
- Real-time weather updates from Open-Meteo

## Files Modified

- `backend/main.py` - Fixed imports, sys.path setup
- `backend/scorer.py` - Added fallback import handling
- `backend/requirements.txt` - Flexible version pinning
- `.gitignore` - Added frontend/node_modules, backend/app.db

## Commits

1. **0975c20** - Add Web UI with conversational agent and RL feedback loop
2. **0be4979** - Fix backend imports and add fallback mode for scoring

## Troubleshooting

**If backend won't start:**
- Ensure Python 3.9+ with scikit-learn 1.9.1
- Check `data/cache/` has all required files
- Verify `outputs/random_forest_model.joblib` exists

**If frontend won't connect to backend:**
- Check backend is on `http://localhost:8000`
- Frontend proxy configured in `vite.config.js`
- No CORS issues (backend has `allow_origins=["http://localhost:5173"]`)

**If agent only gives generic responses:**
- Fallback mode is active (no Anthropic API key)
- Set `ANTHROPIC_API_KEY` env var to enable Claude
- Run `python backend/main.py` to start with Claude enabled

## Research Value

This system demonstrates **online learning in ML recommendation systems**:
- Static baseline: Trained Random Forest
- Adaptive agent: Contextual bandit that learns from user feedback
- Comparison metric: Cumulative regret (static vs adaptive)

Perfect for ML course project on **reinforcement learning + recommendation systems**.
