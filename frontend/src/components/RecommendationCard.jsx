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
      {recommendation.image_url && (
        <div className="card-image">
          <img
            src={recommendation.image_url}
            alt={recommendation.city}
            onError={(e) => {
              e.target.style.display = 'none'
            }}
          />
          {recommendation.image_credit?.photographer && (
            <div className="image-credit">
              Photo by{' '}
              <a href={recommendation.image_credit.photographer_url} target="_blank" rel="noopener noreferrer">
                {recommendation.image_credit.photographer}
              </a>{' '}
              on{' '}
              <a href={recommendation.image_credit.unsplash_url} target="_blank" rel="noopener noreferrer">
                Unsplash
              </a>
            </div>
          )}
        </div>
      )}

      <div className="card-header">
        <h4>
          #{recommendation.rank} {recommendation.city}
          {recommendation.metro && !recommendation.city.startsWith(recommendation.metro) && (
            <span className="metro-label"> · {recommendation.metro} area</span>
          )}
        </h4>
        <div className="score-badge" title={`Predicted rating ${recommendation.rf_score.toFixed(1)}/5`}>
          🎯 {Math.round((recommendation.match_score ?? 0) * 100)}% match
        </div>
      </div>

      {recommendation.reason && <p className="card-reason">{recommendation.reason}</p>}

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
          <span className="stat-label">Budget Fit:</span>
          <div className="stat-bar">
            <div
              className="stat-fill"
              style={{ width: `${(recommendation.budget_match ?? 0.5) * 100}%` }}
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
