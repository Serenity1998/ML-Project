"""Cache and persist processed data and embeddings."""
import json
from pathlib import Path
import pandas as pd
import joblib


class DataCache:
    """Manage cached data and embeddings."""

    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_dataframe(self, df, name):
        """Save dataframe as parquet."""
        path = self.cache_dir / f'{name}.parquet'
        df.to_parquet(path, index=False)
        return path

    def load_dataframe(self, name):
        """Load dataframe from parquet."""
        path = self.cache_dir / f'{name}.parquet'
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_embeddings(self, embeddings, name):
        """Save embeddings as joblib."""
        path = self.cache_dir / f'{name}_embeddings.joblib'
        joblib.dump(embeddings, path)
        return path

    def load_embeddings(self, name):
        """Load embeddings from joblib."""
        path = self.cache_dir / f'{name}_embeddings.joblib'
        if path.exists():
            return joblib.load(path)
        return None

    def save_metadata(self, metadata, name):
        """Save metadata as JSON."""
        path = self.cache_dir / f'{name}_metadata.json'
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return path

    def load_metadata(self, name):
        """Load metadata from JSON."""
        path = self.cache_dir / f'{name}_metadata.json'
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def clear(self):
        """Clear all cached files."""
        for f in self.cache_dir.glob('*'):
            f.unlink()
