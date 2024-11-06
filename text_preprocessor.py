
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def clean_text(self, text):
        # Handle non-string inputs
        if not isinstance(text, str):
            text = '' if pd.isnull(text) else str(text)

        # Lowercase and remove punctuation
        text = text.lower()
        text = ''.join(char for char in text if char.isalnum() or char.isspace())

        # Normalization dictionary
        normalization_dict = {
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

        words = text.split()
        words = [normalization_dict.get(w, w) for w in words]
        return ' '.join(words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply clean_text to each element in X using list comprehension
        return [self.clean_text(text) for text in X]
