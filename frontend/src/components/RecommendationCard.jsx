import { useState } from 'react'
import { submitFeedback } from '../api'
import './RecommendationCard.css'

export default function RecommendationCard({ recommendation, sessionId, onFeedback }) {
  const [feedback, setFeedback] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const handleFeedback = async (reward) => {
    setSubmitting(true)
    try {
      await submitFeedback(sessionId, recommendation.city, reward)
      setFeedback(reward)
      onFeedback()
    } catch (error) {
      console.error('Feedback error:', error)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="recommendation-card">
      <div className="card-header">
        <h4>#{recommendation.rank} {recommendation.city}</h4>
        <div className="score-badge">
          ⭐ {recommendation.rf_score.toFixed(1)}/5
        </div>
      </div>

      <div className="card-stats">
        <div className="stat">
          <span className="stat-label">Climate Match:</span>
          <div className="stat-bar">
            <div
              className="stat-fill"
              style={{ width: `${recommendation.climate_match * 100}%` }}
            />
          </div>
        </div>
        <div className="stat">
          <span className="stat-label">Activity Fit:</span>
          <div className="stat-bar">
            <div
              className="stat-fill"
              style={{ width: `${recommendation.activity_sim * 100}%` }}
            />
          </div>
        </div>
        <div className="stat">
          <span className="stat-label">Popularity:</span>
          <div className="stat-bar">
            <div
              className="stat-fill"
              style={{ width: `${recommendation.popularity * 100}%` }}
            />
          </div>
        </div>
      </div>

      {recommendation.explanation && (
        <p className="explanation">{recommendation.explanation}</p>
      )}

      <div className="feedback-buttons">
        <button
          className={`feedback-btn thumbs-up ${feedback === 1 ? 'active' : ''}`}
          onClick={() => handleFeedback(1)}
          disabled={submitting || feedback !== null}
        >
          👍 Interested
        </button>
        <button
          className={`feedback-btn thumbs-down ${feedback === 0 ? 'active' : ''}`}
          onClick={() => handleFeedback(0)}
          disabled={submitting || feedback !== null}
        >
          👎 Not for me
        </button>
      </div>

      {feedback !== null && (
        <div className="feedback-status">
          ✓ Feedback recorded
        </div>
      )}
    </div>
  )
}
