"""Load and parse Yelp datasets."""
import json
from pathlib import Path
import pandas as pd


def load_json_lines(filepath):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_businesses(filepath):
    """Load Yelp business dataset."""
    return pd.DataFrame(load_json_lines(filepath))


def load_reviews(filepath):
    """Load Yelp review dataset."""
    return pd.DataFrame(load_json_lines(filepath))


def load_users(filepath):
    """Load Yelp user dataset."""
    return pd.DataFrame(load_json_lines(filepath))
