"""
Flask API for destination recommendations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, '.')

from config import *

app = Flask(__name__)
CORS(app)

# Load model and data
print("Loading model...")
rf_model = joblib.load(OUTPUT_DIR / 'random_forest_model_expanded.joblib')
dest_embeddings = joblib.load(OUTPUT_DIR / 'city_embeddings_all.joblib')

# Load embedder
print("Loading embedder...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Interest embeddings
INTERESTS = {
    'beaches': 'beach sand ocean water swimming',
    'nightlife': 'bar club nightlife party dancing',
    'food': 'food restaurant cuisine',
    'culture': 'culture history museum art',
    'nature': 'nature hiking mountain forest',
    'adventure': 'adventure sports extreme',
    'shopping': 'shopping mall retail',
    'relax': 'spa wellness peaceful',
}

interest_embs = {k: embedder.encode(v) for k, v in INTERESTS.items()}

print("✓ Model loaded")
print(f"✓ {len(dest_embeddings)} cities available")
print("✓ API ready!")

@app.route('/api/interests', methods=['GET'])
def get_interests():
    """Return available interests"""
    return jsonify({'interests': list(INTERESTS.keys())})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get recommendations based on preferences"""
    try:
        data = request.json
        interests = data.get('interests', [])

        if not interests:
            return jsonify({'error': 'No interests selected'}), 400

        rankings = []

        for city, dest_emb in dest_embeddings.items():
            # Calculate interest similarity
            sims = []
            for interest in interests:
                if interest in interest_embs:
                    interest_emb = interest_embs[interest]
                    sim = np.dot(dest_emb, interest_emb) / (
                        np.linalg.norm(dest_emb) * np.linalg.norm(interest_emb) + 1e-6
                    )
                    sims.append(sim)

            interest_score = np.mean(sims) if sims else 0.0

            # 90% interests, 10% baseline
            final_score = interest_score * 0.9 + 0.5 * 0.1

            rankings.append({
                'city': city,
                'score': float(final_score),
                'interest_match': float(interest_score)
            })

        # Sort and return top 10
        rankings = sorted(rankings, key=lambda x: x['score'], reverse=True)[:10]

        return jsonify({
            'success': True,
            'interests': interests,
            'recommendations': rankings
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare two preference sets"""
    try:
        data = request.json
        prefs1 = data.get('preferences1', [])
        prefs2 = data.get('preferences2', [])

        def get_top_cities(interests):
            rankings = []
            for city, dest_emb in dest_embeddings.items():
                sims = []
                for interest in interests:
                    if interest in interest_embs:
                        interest_emb = interest_embs[interest]
                        sim = np.dot(dest_emb, interest_emb) / (
                            np.linalg.norm(dest_emb) * np.linalg.norm(interest_emb) + 1e-6
                        )
                        sims.append(sim)

                interest_score = np.mean(sims) if sims else 0.0
                final_score = interest_score * 0.9 + 0.5 * 0.1
                rankings.append({'city': city, 'score': final_score})

            return sorted(rankings, key=lambda x: x['score'], reverse=True)[:5]

        top1 = get_top_cities(prefs1)
        top2 = get_top_cities(prefs2)

        return jsonify({
            'success': True,
            'preferences1': prefs1,
            'recommendations1': top1,
            'preferences2': prefs2,
            'recommendations2': top2
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
