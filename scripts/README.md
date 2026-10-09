# 📁 Scripts Directory

Production-ready Python scripts for the ML recommendation system.

## Pipeline Scripts (Run in Order)

```bash
# Full pipeline with expanded dataset (500+ cities, 2M+ reviews)
python scripts/phase1_expanded.py     # Load & filter data (5 min)
python scripts/phase2_expanded.py     # Analyze destinations (1 min)
python scripts/phase3_expanded.py     # Generate embeddings (20-30 min)
python scripts/phase4_expanded.py     # Train model (5 min)
python scripts/phase5_expanded.py     # Ablation study (5 min)
```

## Utility Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `expand_to_half_cities.py` | Expand dataset to 500+ cities | `python scripts/expand_to_half_cities.py` |
| `app.py` | Flask API backend | `python scripts/app.py` |
| `test_prefs_v2.py` | Test preferences with 90% weighting | `python scripts/test_prefs_v2.py` |
| `count_cities.py` | Count & list all available cities | `python scripts/count_cities.py` |
| `get_all_cities.py` | Extract cities from embeddings | `python scripts/get_all_cities.py` |

## Quick Start for Teammates

**First time setup:**
```bash
cd ML-Project
pip install -r requirements.txt
python scripts/phase1_expanded.py
```

**Run full pipeline:**
```bash
./run_pipeline.sh  # (Linux/Mac)
# or manually run phases 1-5 in order
```

**Start API:**
```bash
python scripts/app.py
# Visit: http://localhost:5000
# Open: index.html in browser
```

## File Organization

```
scripts/
├── README.md (this file)
├── requirements.txt
│
├── Pipeline (run in order):
│  ├── phase1_expanded.py      # Data loading
│  ├── phase2_expanded.py      # Destination analysis
│  ├── phase3_expanded.py      # Embeddings generation
│  ├── phase4_expanded.py      # Model training
│  └── phase5_expanded.py      # Ablation study
│
└── Utilities:
   ├── app.py                  # Flask backend API
   ├── expand_to_half_cities.py # Expand dataset
   ├── test_prefs_v2.py        # Test preferences
   ├── count_cities.py         # Count cities
   └── get_all_cities.py       # Extract cities
```

## Expected Pipeline Output

```
After Phase 5:
  ✓ 2M+ embeddings generated
  ✓ Model trained on 500+ cities
  ✓ Feature importance analyzed
  ✓ Ready for deployment
```

## Troubleshooting

See `EXPANDED_PHASES_README.md` for detailed guide.

Quick issues:
- **Phase 3 slow?** Check GPU: `nvidia-smi`
- **Out of memory?** Reduce batch_size in phase3_expanded.py
- **Missing data?** Download Yelp dataset to `raw_data/`

## Next Steps

1. Run all phases (30-45 min)
2. Start Flask: `python scripts/app.py`
3. Open `index.html` in browser
4. Test recommendations with 500+ cities!

---

**Questions?** Check the docstrings in each script or the main README.
