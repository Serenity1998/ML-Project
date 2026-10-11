import React, { useState, useEffect, useRef } from 'react'
import PreferenceForm from './components/PreferenceForm'
import ChatPanel from './components/ChatPanel'
import PreferenceSummary from './components/PreferenceSummary'
import RecommendationCard from './components/RecommendationCard'
import MetricsPanel from './components/MetricsPanel'
import { chat, getRecommendations, getMetrics, healthCheck } from './api'
import './App.css'

// Older backends return only the flat list; show it as a single section
function toSections(response) {
  return response.sections?.length
    ? response.sections
    : [{ key: 'top', title: '🏙️ Top Recommendations', description: '', recommendations: response.recommendations }]
}

function App() {
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`)
  const [messages, setMessages] = useState([
    { role: 'agent', text: 'Hi! I\'m your travel assistant. Tell me when you want to go and what you enjoy, and I\'ll find destinations, compare them, and adjust them as we chat.' }
  ])
  const [preferences, setPreferences] = useState(null)
  const [chatPrefs, setChatPrefs] = useState(null)  // prefs the assistant searched with; fills the form
  const [recommendations, setRecommendations] = useState(null)
  const [homeInfo, setHomeInfo] = useState(null)  // how the backend resolved "Traveling from"
  const [loading, setLoading] = useState(false)
  const [metrics, setMetrics] = useState(null)
  const [apiHealth, setApiHealth] = useState(null)

  useEffect(() => {
    // Check API health on mount
    healthCheck().then(health => {
      setApiHealth(health.status === 'healthy')
    }).catch(() => {
      setApiHealth(false)
    })

    // Poll metrics every 5 seconds
    const metricsInterval = setInterval(() => {
      getMetrics().then(setMetrics).catch(console.error)
    }, 5000)

    return () => clearInterval(metricsInterval)
  }, [])

  const handleMessage = async (userMessage) => {
    setMessages(prev => [...prev, { role: 'user', text: userMessage }])
    setLoading(true)

    try {
      const response = await chat(sessionId, userMessage)

      setMessages(prev => [...prev, { role: 'agent', text: response.agent_reply }])

      // The assistant ran a search: show its results and mirror its prefs in the form
      if (response.recommendations) {
        setRecommendations(toSections(response.recommendations))
        setHomeInfo(response.recommendations.home)
      }
      if (response.preferences_extracted) {
        setPreferences(response.preferences_extracted)
        setChatPrefs(response.preferences_extracted)
      }
    } catch (error) {
      console.error('Chat error:', error)
      setMessages(prev => [...prev, { role: 'agent', text: 'Sorry, I had trouble processing that. Please try again.' }])
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateRecommendations = async (prefs) => {
    setLoading(true)
    try {
      const response = await getRecommendations(sessionId, prefs)
      setRecommendations(toSections(response))
      setHomeInfo(response.home)
      setPreferences(prefs)
    } catch (error) {
      console.error('Recommendation error:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleFormSubmit = async (formPrefs) => {
    await handleGenerateRecommendations(formPrefs)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>✈️ Travel Destination Recommender</h1>
        <p>ML-powered recommendations with real-time learning</p>
        {apiHealth === false && <div className="warning">⚠️ Backend API not available</div>}
      </header>

      <div className="main-content">
        {/* Form Section */}
        <PreferenceForm onSubmit={handleFormSubmit} loading={loading} syncedPrefs={chatPrefs} homeInfo={homeInfo} />

        {/* Recommendations Section */}
        {recommendations && recommendations.map(section => (
          <div className="recommendations-section" key={section.key}>
            <h2>{section.title}</h2>
            {section.description && <p className="section-description">{section.description}</p>}
            <div className="recommendations-grid">
              {section.recommendations.map(rec => (
                <RecommendationCard
                  key={rec.city}
                  recommendation={rec}
                  sessionId={sessionId}
                  onFeedback={() => {
                    getMetrics().then(setMetrics)
                  }}
                />
              ))}
            </div>
          </div>
        ))}

        {/* Chat & Metrics Section */}
        <div className="bottom-container">
          <div className="chat-section">
            <h2>💬 Chat with Agent (Optional)</h2>
            <ChatPanel
              messages={messages}
              onMessage={handleMessage}
              loading={loading}
            />
          </div>

          {metrics && (
            <div className="metrics-section">
              <MetricsPanel metrics={metrics} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
