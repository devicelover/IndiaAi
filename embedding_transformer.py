# embedding_transformer.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a list of strings
        if isinstance(X, (pd.Series, pd.DataFrame)):
            texts = X.squeeze().tolist()
        elif isinstance(X, np.ndarray):
            texts = X.tolist()
        elif isinstance(X, list):
            texts = X  # X is already a list
        else:
            texts = [str(X)]
        return self.embedding_model.encode(texts, batch_size=32, show_progress_bar=False)
