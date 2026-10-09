# Travel Recommendation System - Complete Overview

## Frontend Architecture

### Components Flow:
```
App.jsx
├── Header (Title + Subtitle)
├── TravelPlannerLayout
    ├── LEFT SIDEBAR (Fixed, Yellow)
    │   ├── ChatPanel (Assistant conversation)
    │   └── SuggestionsPanel (Quick destination list)
    │
    └── RIGHT CONTENT
        ├── Form Section
        │   └── PreferenceForm
        │       ├── Travel Month (select)
        │       ├── Climate Preference (select)
        │       ├── Budget (select)
        │       ├── Activities (checkboxes)
        │       └── Interests (textarea)
        │
        ├── Destination Hero
        │   ├── Hero Image
        │   ├── Title + Description
        │   └── Tabs (Overview, Hotels, Itinerary, Flights)
        │
        ├── Info Cards
        │   ├── Temperature
        │   ├── Price Range
        │   ├── Crowd Level
        │   └── Rating
        │
        └── Feedback Section
            └── RecommendationCard (with 👍/👎 buttons)
```

## Data Flow

### 1. USER SUBMITS PREFERENCES
```
PreferenceForm
    ↓
onFormSubmit (handleGenerateRecommendations)
    ↓
API: POST /recommendations
    ↓
Backend: Generate RF model predictions
    ↓
Return top 10 destinations with:
  - city name
  - rf_score
  - climate_match
  - activity_sim
  - popularity
  - explanation
```

### 2. USER SELECTS DESTINATION
```
User clicks destination in suggestions
    ↓
onDestinationSelect(destination)
    ↓
App state: selectedDestination = destination
    ↓
TravelPlannerLayout renders:
  - DestinationHero
  - InfoCards
  - RecommendationCard (feedback)
```

### 3. USER GIVES FEEDBACK
```
User clicks 👍 or 👎
    ↓
RecommendationCard.handleFeedback(reward)
    ↓
API: POST /feedback
    ↓
Backend: Record user preference
    ↓
Update metrics
    ↓
onFeedback callback → refresh metrics display
```

### 4. USER CHATS WITH AGENT
```
User types message in ChatPanel
    ↓
handleMessage(userMessage)
    ↓
API: POST /chat
    ↓
Backend: Process with conversational agent
    ↓
Extract preferences if mentioned
    ↓
Auto-generate recommendations if prefs extracted
    ↓
Chat updates with agent reply
```

## API Endpoints (Backend)

### Recommendations
**POST /recommendations**
```json
Request: {
  "travel_month": 6,
  "climate_preference": "moderate",
  "activities": ["hiking", "beaches"],
  "budget": "medium",
  "free_text_interests": "nature and food"
}

Response: {
  "recommendations": [
    {
      "rank": 1,
      "city": "Boulder",
      "country": "USA",
      "rf_score": 4.2,
      "climate_match": 0.85,
      "activity_sim": 0.90,
      "popularity": 0.75,
      "explanation": "Great climate for hiking..."
    }
  ]
}
```

### Feedback
**POST /feedback**
```json
Request: {
  "session_id": "session_123456",
  "city": "Boulder",
  "reward": 1  // 1 = interested, 0 = not interested
}

Response: {
  "status": "recorded",
  "metrics": { ... }
}
```

### Chat
**POST /chat**
```json
Request: {
  "session_id": "session_123456",
  "user_message": "I love beaches and warm weather"
}

Response: {
  "agent_reply": "I found some great beach destinations...",
  "preferences_extracted": {
    "climate_preference": "warm",
    "interests": ["beaches"]
  }
}
```

## Feature Pipeline

### PreferenceForm converts to features:
```
User Input:
  - travel_month: 6
  - climate_preference: "moderate"
  - activities: ["hiking", "beaches"]
  - budget: "medium"
  - interests: "nature lover"

↓ PreferenceConverter ↓

Feature Vector (800+ dims):
  - user_embedding (384D): Sentence-BERT of interests
  - user_stats: rating avg, num reviews, etc
  - activity_embeddings: for each activity
  - budget_encoding: low/med/high
  - month_encoding: 1-12

↓ Random Forest Model ↓

Predictions per destination:
  - Predicted rating (0-5)
  - Climate match score (0-1)
  - Activity similarity (0-1)
  - Popularity (0-1)

↓ Rank by score ↓

Top 10 destinations displayed
```

## State Management (App.jsx)

```javascript
const [sessionId]              // Unique session per user
const [messages]               // Chat conversation history
const [preferences]            // Last extracted preferences
const [recommendations]        // List of top 10 recommendations
const [selectedDestination]    // Currently viewing destination
const [metrics]                // RL feedback metrics
const [loading]                // API call state
```

## RL Feedback Loop

```
User rates destination (👍/👎)
    ↓
Store feedback with session ID
    ↓
Accumulate feedback over sessions:
  - Which destinations got thumbs up
  - Which got thumbs down
  - How often recommendations are accepted
    ↓
Metrics tracked:
  - Click-through rate
  - Satisfaction score
  - Preference patterns
    ↓
(Future) Use feedback to:
  - Weight features in model
  - Identify user archetypes
  - Personalize future recommendations
```

## Styling & Layout

### Colors:
- Primary: #6366f1 (Indigo)
- Yellow sidebar: #fef3c7 - #fde68a (gradient)
- Orange chat header: #f59e0b - #d97706
- Grays: #111827 (text) to #f9fafb (bg)

### Responsive Breakpoints:
- Desktop (>1024px): 380px fixed sidebar + flexible content
- Tablet (768-1024px): Stacked layout
- Mobile (<768px): Single column

## Files Structure

```
frontend/src/
├── App.jsx                    # Main app, state management
├── App.css                    # Global styles
├── components/
│   ├── TravelPlannerLayout.jsx    # Main layout component
│   ├── PreferenceForm.jsx         # User preferences form
│   ├── ChatPanel.jsx              # Chat interface
│   ├── DestinationHero.jsx        # Destination hero section
│   ├── InfoCards.jsx              # Weather, price, etc
│   ├── RecommendationCard.jsx     # Feedback buttons
│   ├── SuggestionsPanel.jsx       # Quick destinations
│   └── [component].css            # Component-specific styles
└── api.js                     # API client
```

## How to Test

### 1. Fill Preference Form
- Select travel month
- Choose climate preference
- Pick activities
- Enter interests
- Click "Get Recommendations"

### 2. View Results
- See ranked destinations
- Click on any to view details
- Scroll through info cards

### 3. Give Feedback
- Click 👍 or 👎 on destination
- Feedback saved to backend
- Metrics updated

### 4. Chat with Agent
- Type messages in sidebar
- Agent extracts preferences
- Auto-generates new recommendations

### 5. Monitor Metrics
- View how many feedback entries
- See patterns in preferences
- Check session data

## Next Steps / Improvements

1. **Data Persistence**: Save user sessions to database
2. **RL Fine-tuning**: Use feedback to update model weights
3. **User Profiles**: Track repeat users, learn preferences over time
4. **More Destinations**: Expand beyond 39 US cities
5. **Personalization**: Ensemble models for different user types
6. **A/B Testing**: Test different ranking strategies
7. **Analytics Dashboard**: Visualize feedback patterns
