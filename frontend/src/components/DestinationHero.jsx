import { useState } from 'react'
import './DestinationHero.css'

export default function DestinationHero({ destination, metrics }) {
  const [activeTab, setActiveTab] = useState('overview')

  if (!destination) {
    return null
  }

  const tabs = ['Overview', 'Hotels', 'Itinerary', 'Flights']
  const tabIds = ['overview', 'hotels', 'itinerary', 'flights']

  return (
    <div className="destination-hero">
      <div className="hero-image-wrapper">
        <img
          src={destination.image_url || 'https://via.placeholder.com/800x400?text=' + destination.city}
          alt={destination.city}
          className="hero-image"
        />
        <div className="hero-overlay">
          <div className="hero-content">
            <div className="hero-badge">📍 {destination.country}</div>
            <h1 className="hero-title">{destination.city}</h1>
            <p className="hero-description">
              {destination.description || 'A beautiful destination waiting to be explored'}
            </p>
          </div>
        </div>
      </div>

      <div className="destination-nav">
        {tabs.map((tab, idx) => (
          <button
            key={idx}
            className={`nav-tab ${activeTab === tabIds[idx] ? 'active' : ''}`}
            onClick={() => setActiveTab(tabIds[idx])}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="tab-content">
        {activeTab === 'overview' && (
          <div className="overview-content">
            <h3>About {destination.city}</h3>
            <p>{destination.description || 'Discover the wonders of this amazing destination.'}</p>
          </div>
        )}
        {activeTab === 'hotels' && (
          <div className="tab-content-item">
            <h3>Recommended Hotels</h3>
            <p>Browse top-rated hotels and accommodations for your stay.</p>
          </div>
        )}
        {activeTab === 'itinerary' && (
          <div className="tab-content-item">
            <h3>Sample Itinerary</h3>
            <p>Explore curated activities and attractions.</p>
          </div>
        )}
        {activeTab === 'flights' && (
          <div className="tab-content-item">
            <h3>Flight Options</h3>
            <p>Find the best flight deals and schedules.</p>
          </div>
        )}
      </div>
    </div>
  )
}
