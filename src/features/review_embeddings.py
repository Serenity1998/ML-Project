"""Sentence-BERT review vectors for users and cities (from history reviews only)."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _mean_vectors(texts_df, key, model, max_reviews, random_state):
    """Embed up to max_reviews texts per key and average them into one unit vector per key."""
    sample = (
        texts_df.sample(frac=1, random_state=random_state)
        .groupby(key)
        .head(max_reviews)
    )
    vectors = model.encode(
        sample['text'].str.slice(0, 512).tolist(),
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    means = pd.DataFrame(vectors, index=sample[key].to_numpy()).groupby(level=0).mean()
    return means.div(np.linalg.norm(means.to_numpy(), axis=1), axis=0)


def user_text_vectors(history_reviews, model, max_reviews=10, random_state=42):
    """One vector per user: what they write about places they visited (history only)."""
    return _mean_vectors(history_reviews, 'user_id', model, max_reviews, random_state)


def city_text_vectors(history_reviews, businesses_df, model, max_reviews=200, random_state=42):
    """One vector per city: what people write about it (history only)."""
    texts = history_reviews.merge(businesses_df[['business_id', 'city']], on='business_id')
    return _mean_vectors(texts, 'city', model, max_reviews, random_state)


def text_features(user_vecs, city_vecs, n_components=16, random_state=42):
    """User x city cosine similarity, plus small PCA projections for the Two-Tower inputs."""
    similarity = pd.DataFrame(
        user_vecs.to_numpy() @ city_vecs.to_numpy().T,
        index=user_vecs.index,
        columns=city_vecs.index,
    )
    pca = PCA(n_components=n_components, random_state=random_state).fit(
        np.vstack([user_vecs.to_numpy(), city_vecs.to_numpy()])
    )
    user_pcs = pd.DataFrame(
        pca.transform(user_vecs.to_numpy()), index=user_vecs.index,
        columns=[f'user_pc_{i}' for i in range(n_components)],
    )
    city_pcs = pd.DataFrame(
        pca.transform(city_vecs.to_numpy()), index=city_vecs.index,
        columns=[f'city_pc_{i}' for i in range(n_components)],
    )
    return {'similarity': similarity, 'user_pcs': user_pcs, 'city_pcs': city_pcs}
