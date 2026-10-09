import './MetricsPanel.css'

export default function MetricsPanel({ metrics }) {
  const successRate = metrics.total_feedback > 0
    ? (metrics.total_reward / metrics.total_feedback * 100).toFixed(1)
    : 0

  return (
    <div className="metrics-panel">
      <h3>📊 System Metrics</h3>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{metrics.total_feedback}</div>
          <div className="metric-label">Total Feedback</div>
        </div>

        <div className="metric-card">
          <div className="metric-value">{metrics.total_reward}</div>
          <div className="metric-label">Positive Feedback</div>
        </div>

        <div className="metric-card">
          <div className="metric-value">{successRate}%</div>
          <div className="metric-label">Success Rate</div>
        </div>

        <div className="metric-card">
          <div className="metric-value">{metrics.avg_reward.toFixed(2)}</div>
          <div className="metric-label">Avg Reward</div>
        </div>
      </div>

      <div className="info-box">
        <p>💡 <strong>Reinforcement Learning in Action:</strong> As users provide feedback, the bandit algorithm learns which recommendations work best and adapts over time.</p>
      </div>
    </div>
  )
}
