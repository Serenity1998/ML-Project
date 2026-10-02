"""Utility functions."""
import joblib
from pathlib import Path


def save_model(model, filepath):
    """Save a model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load a model from disk."""
    return joblib.load(filepath)


def save_dataframe(df, filepath):
    """Save a dataframe as parquet."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)


def load_dataframe(filepath):
    """Load a dataframe from parquet."""
    return pd.read_parquet(filepath)
