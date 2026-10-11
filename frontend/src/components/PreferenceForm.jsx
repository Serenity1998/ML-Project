import { useState, useEffect } from 'react'
import './PreferenceForm.css'

const ACTIVITY_OPTIONS = [
  'Hiking', 'Beaches', 'Museums', 'Food/Dining', 'Shopping',
  'Nightlife', 'Parks', 'History', 'Art', 'Sports'
]

// "YYYY-MM-01" for the next time `month` (1-12) comes around
function dateForMonth(month) {
  const today = new Date()
  const year = today.getFullYear() + (month < today.getMonth() + 1 ? 1 : 0)
  return `${year}-${String(month).padStart(2, '0')}-01`
}

export default function PreferenceForm({ onSubmit, loading, syncedPrefs, homeInfo }) {
  // Initialize with a date 6 months from now
  const [travelDate, setTravelDate] = useState(() => dateForMonth(((new Date().getMonth() + 6) % 12) + 1))
  const [climate, setClimate] = useState('moderate')
  const [budget, setBudget] = useState('medium')
  const [activities, setActivities] = useState(['Hiking', 'Beaches'])
  const [interests, setInterests] = useState('')
  const [excludeStates, setExcludeStates] = useState([])
  const [travelingFrom, setTravelingFrom] = useState('')
  const [fromChat, setFromChat] = useState(false)

  // Fill the form with whatever the chat assistant last searched with
  useEffect(() => {
    if (!syncedPrefs) return
    setTravelDate(dateForMonth(syncedPrefs.travel_month))
    setClimate(syncedPrefs.climate_preference)
    setBudget(syncedPrefs.budget)
    setActivities(syncedPrefs.activities || [])
    setInterests(syncedPrefs.free_text_interests || '')
    setExcludeStates(syncedPrefs.exclude_states || [])
    setTravelingFrom(syncedPrefs.traveling_from || '')
    setFromChat(true)
  }, [syncedPrefs])

  const toggleActivity = (activity) => {
    setActivities(prev =>
      prev.includes(activity)
        ? prev.filter(a => a !== activity)
        : [...prev, activity]
    )
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    // Month (1-12) straight from "YYYY-MM-DD" (new Date() would parse it as UTC
    // and can land on the previous month in western time zones)
    const month = parseInt(travelDate.split('-')[1], 10)
    setFromChat(false)

    onSubmit({
      travel_month: month,
      climate_preference: climate,
      activities: activities,
      budget: budget,
      free_text_interests: interests,
      exclude_states: excludeStates,
      traveling_from: travelingFrom.trim()
    })
  }

  return (
    <form className="preference-form" onSubmit={handleSubmit} onChange={() => setFromChat(false)}>
      {fromChat && <div className="synced-note">✨ Filled in from your chat with the assistant</div>}
      <div className="form-grid">
        <div className="form-group">
          <label htmlFor="travelDate">Travel Date</label>
          <input
            id="travelDate"
            type="date"
            value={travelDate}
            onChange={(e) => setTravelDate(e.target.value)}
            disabled={loading}
          />
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
        <label htmlFor="travelingFrom">Traveling from (optional)</label>
        <input
          id="travelingFrom"
          type="text"
          value={travelingFrom}
          onChange={(e) => setTravelingFrom(e.target.value)}
          placeholder="E.g., Chicago or Edmonton"
          maxLength={80}
          disabled={loading}
        />
        {homeInfo?.query && (
          <div className={`home-note ${homeInfo.found ? '' : 'home-note-missing'}`}>
            {homeInfo.found
              ? `📍 From ${homeInfo.label}. ${homeInfo.nearby_excluded
                  ? `Left out ${homeInfo.nearby_excluded} ${homeInfo.nearby_excluded === 1 ? 'city' : 'cities'} within ${homeInfo.min_trip_km} km.`
                  : `No cities within ${homeInfo.min_trip_km} km to leave out.`}`
              : `Couldn't find "${homeInfo.query}", so showing results without it.`}
          </div>
        )}
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

      {excludeStates.length > 0 && (
        <div className="excluded-states">
          Excluding: {excludeStates.join(', ')}
          <button type="button" onClick={() => setExcludeStates([])} disabled={loading}>Clear</button>
        </div>
      )}

      <button type="submit" className="submit-btn" disabled={loading}>
        {loading ? 'Finding recommendations...' : 'Get Recommendations'}
      </button>
    </form>
  )
}
