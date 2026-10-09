import './InfoCards.css'

export default function InfoCards({ destination, metrics }) {
  const cards = [
    {
      id: 'temperature',
      icon: '🌡️',
      label: 'Temperature',
      value: destination?.temperature ? `${destination.temperature}°C` : '28°C',
      subtext: 'for your dates'
    },
    {
      id: 'pricing',
      icon: '💰',
      label: 'Price Range',
      value: destination?.price_range || '+1,591 - 13,162',
      subtext: 'estimated per night'
    },
    {
      id: 'crowds',
      icon: '👥',
      label: 'Crowds',
      value: destination?.crowd_level || 'Not to busy',
      subtext: 'current forecast'
    },
    {
      id: 'rating',
      icon: '⭐',
      label: 'Rating',
      value: destination?.rating || '4.8/5',
      subtext: 'from travelers'
    }
  ]

  return (
    <div className="info-cards">
      {cards.map(card => (
        <div key={card.id} className="info-card">
          <div className="card-icon">{card.icon}</div>
          <div className="card-content">
            <div className="card-label">{card.label}</div>
            <div className="card-value">{card.value}</div>
            <div className="card-subtext">{card.subtext}</div>
          </div>
        </div>
      ))}
    </div>
  )
}
