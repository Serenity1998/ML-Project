# Web UI & Conversational Agent Setup Guide

This guide explains how to run the new **FastAPI backend** + **React frontend** with a **Claude-powered conversational agent** and **RL feedback loop**.

## Architecture Overview

- **Backend**: FastAPI server with Random Forest model inference, contextual bandit learning, Claude agent integration
- **Frontend**: React (Vite) chat interface with recommendation cards and metrics dashboard
- **Agent**: Claude Haiku via Anthropic API (or rule-based fallback if no key)
- **Learning**: Lin-TS contextual bandit that learns from user feedback (thumbs up/down)

## Prerequisites

- Python 3.9+
- Node.js 16+ and npm
- Anthropic API key (optional, for Claude agent; fallback available)

## Backend Setup

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note:** This pins `scikit-learn==1.9.1` (matching the model training environment).

### 2. Set Environment Variables

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or on Windows (PowerShell):
```powershell
$env:ANTHROPIC_API_KEY = "your-api-key-here"
```

### 3. Verify Data Files

The backend expects these cached files in `data/cache/`:
- `destination_embeddings.joblib` (39 cities × 384 dims)
- `weather_features_for_models.parquet` (city weather data)
- `major_cities_metadata.json` (list of 39 candidate cities)
- `businesses_major_cities.parquet` (for computing city scalars)
- `train_reviews.parquet` (for computing city statistics)

If any are missing, run the relevant phase notebooks to regenerate them.

### 4. Start Backend Server

```bash
python main.py
```

or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be at `http://localhost:8000`.

**Health check:**
```bash
curl http://localhost:8000/health
```

## Frontend Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Dev Server

```bash
npm run dev
```

Frontend will be at `http://localhost:5173`.

### 3. Configure API Proxy

The Vite config (`vite.config.js`) already proxies `/api/*` to `http://localhost:8000`. 

The React API client (`src/api.js`) calls the backend directly on `http://localhost:8000`.

## How It Works

### User Flow

1. **Chat Phase**: User converses with Claude agent about travel preferences
2. **Extraction**: Agent extracts structured preferences (month, climate, activities, budget)
3. **Recommendation**: Backend scores all 39 cities using Random Forest + contextual bandit
4. **Feedback**: User gives thumbs up/down on recommendations
5. **Learning**: Bandit updates its model from feedback, improving future recommendations

### Conversational Agent

- Uses Claude Haiku to have natural conversations
- Asks about: travel month, climate preference, activities, budget, free-text interests
- Extracts JSON when preferences are complete
- Explains recommendations using city features and model reasoning

**Fallback**: If no `ANTHROPIC_API_KEY`, uses rule-based agent with hardcoded questions.

### RL Feedback Loop

- **Algorithm**: Linear Thompson Sampling (LinTS)
- **Context**: [RF score, climate match, activity similarity, popularity]
- **Reward**: 1 for thumbs up, 0 for thumbs down
- **Learning**: Updates shared model across all users (contextual generalization)
- **Persistence**: State saved to SQLite (`backend/app.db`)

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Check server + model health |
| POST | `/chat` | Send message to agent, get reply + extracted prefs |
| POST | `/recommend` | Get ranked recommendations for user prefs |
| POST | `/feedback` | Log thumbs up/down on a city |
| GET | `/metrics` | Get system metrics (total feedback, success rate) |

## Testing

### Backend Only

```bash
# In backend directory
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "I love warm weather"}'
```

### Full Flow (Browser)

1. Open `http://localhost:5173`
2. Chat with the agent: "I want to travel in summer, I love hiking"
3. Review extracted preferences
4. See recommendations
5. Give feedback (👍/👎)
6. Watch metrics update

## Troubleshooting

### "ModuleNotFoundError: No module named 'anthropic'"

Install the backend dependencies:
```bash
pip install -r backend/requirements.txt
```

### "RandomForestRecommender not found"

Ensure the project root is in `PYTHONPATH`:
```bash
cd backend
export PYTHONPATH="..:$PYTHONPATH"
python main.py
```

### "CORS error in frontend"

Check that backend is running on port 8000. The frontend (`localhost:5173`) is configured to proxy to it.

### Agent giving generic responses

If `ANTHROPIC_API_KEY` is not set, the fallback agent activates. Set the key to use Claude:
```bash
export ANTHROPIC_API_KEY="sk-..."
```

### SQLite database locked

If running multiple backend instances, they'll conflict on `backend/app.db`. Use a single instance or delete the db and restart.

## Next Steps

### Evaluation

Run the simulated-user harness to compare bandit vs static RF:
```bash
python backend/simulate.py
```

This generates synthetic user preferences and measures cumulative reward over time.

### Deployment

For production:
1. Build frontend: `npm run build` → `dist/` folder
2. Serve frontend from CDN or same server as backend
3. Use a production ASGI server: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app`
4. Use PostgreSQL instead of SQLite for bandit state
5. Add authentication and rate limiting

### Extensions

- Add more features to recommendation context (e.g., user profile embeddings)
- Integrate with real booking APIs (Airbnb, Expedia)
- Add seasonal/real-time weather updates
- Multi-armed bandit variants (Thompson Sampling, LinUCB, etc.)

## References

- **Random Forest Model**: `outputs/random_forest_model.joblib` (trained in Phase 4)
- **City Data**: `data/cache/major_cities_metadata.json` (39 US cities)
- **Weather Data**: Open-Meteo historical climate data
- **Embeddings**: SentenceTransformers `all-MiniLM-L6-v2`
- **Agent**: Anthropic Claude Haiku API
