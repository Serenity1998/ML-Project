import os
import json
from typing import Optional, Dict, Any
from anthropic import Anthropic
import re

# Initialize Anthropic client
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
client = Anthropic() if API_KEY else None

SYSTEM_PROMPT = """You are a friendly travel recommendation assistant. Your job is to:
1. Ask the user about their travel preferences (month, climate, activities, budget, interests)
2. Extract structured preferences from the conversation
3. Once you have gathered enough information, summarize the preferences and ask for confirmation

Keep responses conversational and friendly. After gathering preferences, output a JSON block with the extracted prefs:

```json
{
  "travel_month": 6,
  "climate_preference": "warm",
  "activities": ["hiking", "museums"],
  "budget": "medium",
  "free_text_interests": "I love outdoor activities and good food"
}
```

Only output this JSON when you have enough info to make recommendations."""

def extract_prefs_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract structured preferences from agent response."""
    # Look for JSON block in the response
    json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    return None

def chat_with_agent(session_id: str, user_message: str, conversation_history: list) -> tuple[str, Optional[Dict]]:
    """
    Send message to Claude agent and get response.

    Returns:
        (agent_response, extracted_prefs_or_None)
    """

    # Fallback if no API key
    if not client:
        response = fallback_agent_response(user_message, conversation_history)
        return response, None

    # Add user message to history
    messages = conversation_history + [{"role": "user", "content": user_message}]

    # Call Claude
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    agent_text = response.content[0].text

    # Try to extract preferences
    prefs = extract_prefs_from_response(agent_text)

    return agent_text, prefs

def fallback_agent_response(user_message: str, history: list) -> str:
    """Rule-based fallback when no API key."""

    # Simple state machine for gathering prefs
    turn = len(history) // 2

    if turn == 0:
        return "Hi! I'd love to help you find an amazing travel destination. What month are you thinking of traveling?"
    elif turn == 1:
        return "Great! What kind of climate do you prefer? (warm, cool, moderate)"
    elif turn == 2:
        return "Nice! What activities interest you? (e.g., hiking, museums, beaches, food)"
    elif turn == 3:
        return "What's your budget level? (low, medium, high)"
    elif turn == 4:
        return f"""Perfect! Based on our chat, here are your preferences:
- Travel month: June (default)
- Climate: moderate
- Activities: outdoor, cultural
- Budget: medium
- Interests: {user_message[:50]}

Let me find the best destinations for you!

```json
{
  "travel_month": 6,
  "climate_preference": "moderate",
  "activities": ["hiking", "museums"],
  "budget": "medium",
  "free_text_interests": "outdoor adventures and cultural experiences"
}
```"""
    else:
        return "Thanks! I'll recommend some cities based on your preferences."

def explain_recommendations(city_data: list, user_prefs: Dict) -> Dict[str, str]:
    """
    Generate explanations for top recommendations using Claude.

    Args:
        city_data: list of dicts with city info (name, rf_score, climate_match, etc.)
        user_prefs: user preference dict

    Returns:
        dict mapping city name -> explanation string
    """

    if not client or not city_data:
        # Fallback explanations
        return {
            city["city"]: f"Great match for your {user_prefs.get('climate_preference', 'travel')} preference!"
            for city in city_data[:5]
        }

    explanations = {}
    for city_info in city_data[:5]:  # Top 5 only
        city = city_info["city"]
        prompt = f"""Briefly (1-2 sentences) explain why {city} is a great travel destination for someone who:
- Traveling in month {user_prefs.get('travel_month', 6)}
- Prefers {user_prefs.get('climate_preference', 'moderate')} climate
- Interested in: {', '.join(user_prefs.get('activities', ['general travel']))}
- Budget: {user_prefs.get('budget', 'medium')}

City stats: Rating {city_info.get('rf_score', 3.5):.1f}/5, {city_info.get('dest_num_reviews', 0)} reviews"""

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        explanations[city] = response.content[0].text

    return explanations
