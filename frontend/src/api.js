const API_BASE = 'http://localhost:8001'

export async function chat(sessionId, message) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message })
  })
  return response.json()
}

export async function getRecommendations(sessionId, preferences) {
  const response = await fetch(`${API_BASE}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      preferences
    })
  })
  return response.json()
}

export async function submitFeedback(sessionId, city, reward) {
  const response = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, city, reward })
  })
  return response.json()
}

export async function getMetrics() {
  const response = await fetch(`${API_BASE}/metrics`)
  return response.json()
}

export async function healthCheck() {
  try {
    const response = await fetch(`${API_BASE}/health`)
    if (!response.ok) {
      console.error(`Health check failed: ${response.status} ${response.statusText}`)
      throw new Error(`HTTP ${response.status}`)
    }
    const data = await response.json()
    console.log('Health check success:', data)
    return data
  } catch (error) {
    console.error('Health check error:', error.message, error)
    throw error
  }
}
