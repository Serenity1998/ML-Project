"""Load and parse Yelp datasets."""
import json
from pathlib import Path
import pandas as pd
import random


def load_json_lines(filepath):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json_lines_sample(filepath, sample_fraction=0.1, random_state=42):
    """Load stratified sample of JSONL file."""
    random.seed(random_state)
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and random.random() < sample_fraction:
                data.append(json.loads(line))
    return data


def load_businesses(filepath, sample=False, sample_fraction=0.1):
    """Load Yelp business dataset."""
    if sample:
        data = load_json_lines_sample(filepath, sample_fraction)
    else:
        data = load_json_lines(filepath)
    return pd.DataFrame(data)


def load_reviews(filepath, sample=False, sample_fraction=0.1):
    """Load Yelp review dataset."""
    if sample:
        data = load_json_lines_sample(filepath, sample_fraction)
    else:
        data = load_json_lines(filepath)
    return pd.DataFrame(data)


def load_users(filepath, sample=False, sample_fraction=0.1):
    """Load Yelp user dataset."""
    if sample:
        data = load_json_lines_sample(filepath, sample_fraction)
    else:
        data = load_json_lines(filepath)
    return pd.DataFrame(data)


def load_tips(filepath, sample=False, sample_fraction=0.1):
    """Load Yelp tips dataset."""
    if sample:
        data = load_json_lines_sample(filepath, sample_fraction)
    else:
        data = load_json_lines(filepath)
    return pd.DataFrame(data)
