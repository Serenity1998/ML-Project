"""Load and parse Yelp datasets."""
import hashlib
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


def is_user_sampled(user_id, sample_fraction=0.1, seed=42):
    """Deterministically decide if a user is in the sample (stable across files and runs)."""
    digest = hashlib.md5(f'{seed}:{user_id}'.encode('utf-8')).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < sample_fraction


def load_json_lines_user_sample(filepath, sample_fraction=0.1, seed=42):
    """Load all records of a sampled subset of users from a JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if is_user_sampled(record['user_id'], sample_fraction, seed):
                    data.append(record)
    return data


def load_reviews_by_user_sample(filepath, sample_fraction=0.1, seed=42):
    """Load every review written by a sampled subset of users."""
    return pd.DataFrame(load_json_lines_user_sample(filepath, sample_fraction, seed))


def load_users_by_user_sample(filepath, sample_fraction=0.1, seed=42):
    """Load the same sampled subset of users as load_reviews_by_user_sample."""
    return pd.DataFrame(load_json_lines_user_sample(filepath, sample_fraction, seed))


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
