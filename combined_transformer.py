# combined_transformer.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import re

class CombinedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(self.model_name)
        self.normalization_dict = {
            'kyu': 'why',
            'hai': 'is',
            'nahi': 'no',
            'kaise': 'how',
            'kya': 'what',
            'mein': 'in',
            'ho': 'is',
            'raha': 'going',
            'hu': 'am',
            'hindi': 'hindi',
            'english': 'english',
            'agar': 'if',
            'karna': 'do',
            'kar': 'do',
            'ke': 'of',
            'liye': 'for',
            'mera': 'my',
            'tera': 'your',
            'unka': 'their',
            'woh': 'they',
            'bata': 'tell',
            'pata': 'know',
            'sath': 'with',
            'tak': 'until',
            'ab': 'now',
            'sab': 'all',
            'jaldi': 'quickly',
            'kam': 'work',
            'paise': 'money',
            'aap': 'you',
            'tum': 'you',
            'kyunki': 'because',
            'kuch': 'some',
            'ja': 'go',
            'lo': 'take',
            'de': 'give',
            'mujhe': 'me',
            'bhi': 'also',
            'baat': 'talk',
            'matlab': 'meaning',
            'galat': 'wrong',
            'sahi': 'right',
            'phone': 'phone',
            'chori': 'theft',
            'fraud': 'fraud',
            'online': 'online',
            'dhokha': 'cheating',
            'saath': 'with',
            'ghar': 'home',
            'jagah': 'place',
            'sudhar': 'improve',
            'karo': 'do',
            'baar': 'time',
            'kaam': 'job',
            'zindagi': 'life',
            'band': 'closed',
            'chalu': 'open',
            'aata': 'comes',
            'gaya': 'gone',
            'shuru': 'start',
            'khatam': 'end',
            'samajh': 'understand',
            'tarah': 'way',
            'dost': 'friend',
            'roko': 'stop',
            'soch': 'think',
            'dikkat': 'problem',
            'madad': 'help',
            'barabar': 'equal',
            'jhoot': 'lie',
            'asli': 'real',
            'nakli': 'fake',
            'baaki': 'remaining',
            'pura': 'complete',
            'aadmi': 'man',
            'aurat': 'woman',
            'bacha': 'child',
            'police': 'police',
            'report': 'report',
            'shikayat': 'complaint',
            'puri': 'full',
            'naya': 'new',
            'purana': 'old',
            'chal': 'move',
            'ruk': 'stop',
            'faida': 'benefit',
            'nuksan': 'loss',
            'samasya': 'issue',
            'bade': 'big',
            'chote': 'small',
            'dheere': 'slow',
            'yahan': 'here',
            'wahan': 'there',
            'kal': 'yesterday/tomorrow',
            'aaj': 'today',
            'bahut': 'very',
            'sabse': 'most',
            'kamzor': 'weak',
            'majboot': 'strong',
            'samay': 'time',
            'muft': 'free',
            'sasta': 'cheap',
            'mehnga': 'expensive',
            'bhool': 'forget',
            'yaad': 'remember',
            'saaf': 'clear',
            'ganda': 'dirty',
            'zinda': 'alive',
            'marta': 'dying'
        }

    def clean_texts(self, texts):
        # Convert to lowercase
        texts = texts.str.lower()
        # Remove punctuation and non-alphanumeric characters
        texts = texts.apply(lambda x: re.sub(r'[^\w\s]', '', x))
        # Normalize words
        texts = texts.apply(
            lambda x: ' '.join(self.normalization_dict.get(word, word) for word in x.split())
        )
        return texts

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            texts = X.fillna('')
        else:
            texts = pd.Series(X).fillna('')
        cleaned_texts = self.clean_texts(texts)
        embeddings = self.embedding_model.encode(
            cleaned_texts.tolist(), batch_size=64, show_progress_bar=False
        )
        return embeddings
