import { useState } from 'react'
import ChatPanel from './ChatPanel'
import DestinationHero from './DestinationHero'
import InfoCards from './InfoCards'
import PreferenceForm from './PreferenceForm'
import RecommendationCard from './RecommendationCard'
import './TravelPlannerLayout.css'

export default function TravelPlannerLayout({
  messages,
  onMessage,
  loading,
  destination,
  metrics,
  recommendations = [],
  onDestinationSelect,
  onFormSubmit,
  sessionId,
  onFeedback
}) {
  const handleDestinationSelect = (dest) => {
    if (onDestinationSelect) {
      onDestinationSelect(dest)
    }
  }

  return (
    <div className="travel-planner-layout">
      {/* Left Sidebar - Chat */}
      <div className="planner-sidebar">
        <div className="chat-wrapper">
          <div className="chat-header">
            <div className="assistant-badge">
              <span className="badge-icon">🤖</span>
              <span>Assistant</span>
            </div>
          </div>
          <ChatPanel
            messages={messages}
            onMessage={onMessage}
            loading={loading}
          />
        </div>

        {/* Destination Suggestions */}
        {recommendations.length > 0 && (
          <div className="suggestions-panel">
            <h4>Quick Destinations</h4>
            <div className="suggestion-list">
              {recommendations.slice(0, 3).map((rec, idx) => (
                <div
                  key={idx}
                  className={`suggestion-item ${destination?.city === rec.city ? 'active' : ''}`}
                  onClick={() => handleDestinationSelect(rec)}
                >
                  <div className="suggestion-flag">{rec.country_flag || '🌍'}</div>
                  <div className="suggestion-info">
                    <div className="suggestion-city">{rec.city}</div>
                    <div className="suggestion-country">{rec.country}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right Content - Form & Destination Details */}
      <div className="planner-content">
        <div className="form-section">
          <PreferenceForm
            onSubmit={onFormSubmit}
            loading={loading}
          />
        </div>

        {destination ? (
          <>
            <DestinationHero destination={destination} metrics={metrics} />
            <div className="info-section">
              <InfoCards destination={destination} metrics={metrics} />
            </div>
            {sessionId && (
              <div className="feedback-section">
                <h3>How do you like this destination?</h3>
                <RecommendationCard
                  recommendation={destination}
                  sessionId={sessionId}
                  onFeedback={onFeedback || (() => {})}
                />
              </div>
            )}
          </>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">✈️</div>
            <h2>Plan Your Next Trip</h2>
            <p>Fill in your preferences above to get personalized recommendations</p>
          </div>
        )}
      </div>
    </div>
  )
}
