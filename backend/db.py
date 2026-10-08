import sqlite3
from pathlib import Path
from datetime import datetime
import json

DB_PATH = Path(__file__).parent / "app.db"

def init_db():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_prefs TEXT,
            message_count INTEGER DEFAULT 0
        )
    """)

    # Feedback table (for tracking user feedback on recommendations)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            city TEXT NOT NULL,
            reward INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    # Bandit state table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bandit_state (
            id INTEGER PRIMARY KEY,
            V TEXT NOT NULL,
            b TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Metrics table (for aggregated stats)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_feedback INTEGER DEFAULT 0,
            bandit_wins INTEGER DEFAULT 0,
            rf_wins INTEGER DEFAULT 0,
            cumulative_reward REAL DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()

def save_session(session_id: str, user_prefs: dict = None):
    """Save or update a user session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    prefs_json = json.dumps(user_prefs) if user_prefs else None

    cursor.execute("""
        INSERT OR REPLACE INTO sessions (session_id, user_prefs)
        VALUES (?, ?)
    """, (session_id, prefs_json))

    conn.commit()
    conn.close()

def get_session(session_id: str) -> dict:
    """Retrieve session data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT session_id, created_at, user_prefs, message_count
        FROM sessions
        WHERE session_id = ?
    """, (session_id,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "session_id": row[0],
            "created_at": row[1],
            "user_prefs": json.loads(row[2]) if row[2] else None,
            "message_count": row[3]
        }
    return None

def log_feedback(session_id: str, city: str, reward: int):
    """Log user feedback (thumbs up/down) on a city recommendation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO feedback (session_id, city, reward)
        VALUES (?, ?, ?)
    """, (session_id, city, reward))

    conn.commit()
    conn.close()

def get_feedback(session_id: str) -> list:
    """Get all feedback for a session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT city, reward, timestamp
        FROM feedback
        WHERE session_id = ?
        ORDER BY timestamp DESC
    """, (session_id,))

    rows = cursor.fetchall()
    conn.close()

    return [{"city": r[0], "reward": r[1], "timestamp": r[2]} for r in rows]

def get_metrics() -> dict:
    """Get overall metrics for bandit vs static RF."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Count total feedback
    cursor.execute("SELECT COUNT(*), SUM(reward) FROM feedback")
    total_feedback, total_reward = cursor.fetchone()

    conn.close()

    return {
        "total_feedback": total_feedback or 0,
        "total_reward": total_reward or 0,
        "avg_reward": (total_reward or 0) / max(total_feedback or 1, 1)
    }
