import './PreferenceSummary.css'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export default function PreferenceSummary({ preferences }) {
  return (
    <div className="preference-summary">
      <h3>📋 Your Preferences</h3>
      <div className="prefs-grid">
        <div className="pref-item">
          <span className="label">Travel Month:</span>
          <span className="value">{MONTHS[preferences.travel_month - 1]}</span>
        </div>
        <div className="pref-item">
          <span className="label">Climate:</span>
          <span className="value">{preferences.climate_preference}</span>
        </div>
        <div className="pref-item">
          <span className="label">Budget:</span>
          <span className="value">{preferences.budget}</span>
        </div>
        <div className="pref-item">
          <span className="label">Activities:</span>
          <div className="activities">
            {preferences.activities.map(activity => (
              <span key={activity} className="activity-tag">{activity}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
