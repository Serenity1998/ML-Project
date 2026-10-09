import { useState } from 'react'
import './PreferenceForm.css'

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']

const ACTIVITY_OPTIONS = [
  'Hiking', 'Beaches', 'Museums', 'Food/Dining', 'Shopping',
  'Nightlife', 'Parks', 'History', 'Art', 'Sports'
]

export default function PreferenceForm({ onSubmit, loading }) {
  const [month, setMonth] = useState(6)
  const [climate, setClimate] = useState('moderate')
  const [budget, setBudget] = useState('medium')
  const [activities, setActivities] = useState(['Hiking', 'Beaches'])
  const [interests, setInterests] = useState('')

  const toggleActivity = (activity) => {
    setActivities(prev =>
      prev.includes(activity)
        ? prev.filter(a => a !== activity)
        : [...prev, activity]
    )
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit({
      travel_month: month,
      climate_preference: climate,
      activities: activities,
      budget: budget,
      free_text_interests: interests
    })
  }

  return (
    <form className="preference-form" onSubmit={handleSubmit}>
      <div className="form-grid">
        <div className="form-group">
          <label htmlFor="month">Travel Month</label>
          <select
            id="month"
            value={month}
            onChange={(e) => setMonth(parseInt(e.target.value))}
            disabled={loading}
          >
            {MONTHS.map((m, i) => (
              <option key={i} value={i + 1}>{m}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="climate">Climate Preference</label>
          <select
            id="climate"
            value={climate}
            onChange={(e) => setClimate(e.target.value)}
            disabled={loading}
          >
            <option value="warm">Warm</option>
            <option value="moderate">Moderate</option>
            <option value="cool">Cool</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="budget">Budget</label>
          <select
            id="budget"
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
            disabled={loading}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
      </div>

      <div className="form-group">
        <label>Interests & Activities</label>
        <div className="activities-grid">
          {ACTIVITY_OPTIONS.map(activity => (
            <label key={activity} className="activity-checkbox">
              <input
                type="checkbox"
                checked={activities.includes(activity)}
                onChange={() => toggleActivity(activity)}
                disabled={loading}
              />
              <span>{activity}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="form-group">
        <label htmlFor="interests">Tell us more about your interests</label>
        <textarea
          id="interests"
          value={interests}
          onChange={(e) => setInterests(e.target.value)}
          placeholder="E.g., I love outdoor activities, good food, and cultural experiences..."
          rows={4}
          disabled={loading}
        />
      </div>

      <button type="submit" className="submit-btn" disabled={loading}>
        {loading ? 'Finding recommendations...' : 'Get Recommendations'}
      </button>
    </form>
  )
}
