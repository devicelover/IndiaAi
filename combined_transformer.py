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
    'kyu': 'why', 'ky': 'why', 'kyo': 'why', 'kuy': 'why', 'hai': 'is', 'ha': 'is', 'h': 'is',
    'nhi': 'no', 'nahi': 'no', 'nahin': 'no', 'nah': 'no', 'kaise': 'how', 'kaisee': 'how', 'kese': 'how',
    'kaisa': 'how', 'kya': 'what', 'kia': 'what', 'ky': 'what', 'kyya': 'what', 'koe': 'anyone',
    'koi': 'anyone', 'kuch': 'some', 'kch': 'some', 'kucch': 'some', 'mein': 'in', 'mai': 'in',
    'me': 'in', 'ho': 'is', 'hoon': 'am', 'hu': 'am', 'hun': 'am', 'hindhi': 'hindi', 'hindi': 'hindi',
    'hnd': 'hindi', 'angrezi': 'english', 'eng': 'english', 'ang': 'english', 'agar': 'if', 'agr': 'if',
    'ager': 'if', 'karna': 'do', 'krna': 'do', 'kr': 'do', 'kar': 'do', 'kre': 'do', 'ke': 'of', 'k': 'of',
    'lye': 'for', 'liye': 'for', 'liye': 'for', 'mera': 'my', 'meri': 'my', 'mra': 'my', 'ter': 'your',
    'tera': 'your', 'tere': 'your', 'tumhara': 'your', 'tumhare': 'your', 'unka': 'their', 'unke': 'their',
    'unk': 'their', 'wo': 'they', 'woah': 'they', 'woh': 'they', 'bataya': 'told', 'btaya': 'told',
    'batana': 'tell', 'bta': 'tell', 'bata': 'tell', 'pata': 'know', 'ptha': 'know', 'pata': 'know',
    'sath': 'with', 'saath': 'with', 'sathh': 'with', 'saathh': 'with', 'takk': 'until', 'tak': 'until',
    'ab': 'now', 'sab': 'all', 'sbb': 'all', 'sabhi': 'everyone', 'sbhi': 'everyone', 'jldi': 'quickly',
    'jaldi': 'quickly', 'km': 'work', 'kam': 'work', 'kaam': 'job', 'paisa': 'money', 'pese': 'money',
    'aap': 'you', 'app': 'you', 'ap': 'you', 'tum': 'you', 'tu': 'you', 'ty': 'you', 'kyunki': 'because',
    'kuki': 'because', 'kyuki': 'because', 'j': 'go', 'jaa': 'go', 'lo': 'take', 'de': 'give', 'dediya': 'given',
    'dedo': 'give', 'mujhe': 'me', 'mjh': 'me', 'mjhe': 'me', 'mujh': 'me', 'bhi': 'also', 'bhae': 'brother',
    'bhaiya': 'brother', 'bhayo': 'brothers', 'bhaiyo': 'brothers', 'baat': 'talk', 'bat': 'talk',
    'matlab': 'meaning', 'mtlb': 'meaning', 'mltb': 'meaning', 'glat': 'wrong', 'galat': 'wrong',
    'galath': 'wrong', 'phn': 'phone', 'fone': 'phone', 'fon': 'phone', 'fraud': 'fraud', 'frd': 'fraud',
    'dhoka': 'cheating', 'dhokha': 'cheating', 'saath': 'with', 'ghar': 'home', 'jga': 'place', 'sudar': 'improve',
    'sudhar': 'improve', 'karo': 'do', 'krdo': 'do', 'kardo': 'do', 'baar': 'time', 'bar': 'time',
    'zindgee': 'life', 'zindagi': 'life', 'zindagi': 'life', 'bnd': 'closed', 'band': 'closed', 'chl': 'open',
    'chalu': 'open', 'chlau': 'open', 'ata': 'comes', 'aata': 'comes', 'aya': 'comes', 'gya': 'gone',
    'gya': 'gone', 'khtam': 'end', 'khatam': 'end', 'smjh': 'understand', 'samaj': 'understand',
    'samajh': 'understand', 'tarah': 'way', 'dost': 'friend', 'dosth': 'friend', 'frnd': 'friend', 'frndz': 'friends',
    'soch': 'think', 'dkt': 'problem', 'madat': 'help', 'mdad': 'help', 'jhooth': 'lie', 'jhoot': 'lie',
    'asli': 'real', 'aadi': 'half', 'aadmi': 'man', 'mrd': 'man', 'aurat': 'woman', 'orath': 'woman',
    'bacha': 'child', 'childr': 'children', 'bacho': 'children', 'shikayat': 'complaint', 'complain': 'complaint',
    'poorna': 'complete', 'naya': 'new', 'purana': 'old', 'jao': 'go', 'jayega': 'will go', 'bar': 'time',
    'faida': 'benefit', 'bhai': 'brother', 'faida': 'benefit', 'faida': 'benefit', 'badi': 'big',
    'bade': 'big', 'chota': 'small', 'kma': 'earn', 'deere': 'slow', 'dhire': 'slow', 'yaha': 'here',
    'yahan': 'here', 'waha': 'there', 'kal': 'yesterday', 'kal': 'tomorrow', 'bahut': 'very', 'bhut': 'very',
    'sbse': 'most', 'kmzor': 'weak', 'majboot': 'strong', 'samay': 'time', 'smye': 'time', 'muft': 'free',
    'sasta': 'cheap', 'mahnga': 'expensive', 'mehnga': 'expensive', 'bhool': 'forget', 'bhul': 'forget',
    'yad': 'remember', 'yaad': 'remember', 'saaf': 'clear', 'clear': 'clear', 'gnda': 'dirty', 'zinda': 'alive',
    'mt': 'not', 'otp': 'OTP', 'suchna': 'inform', 'inform': 'inform', 'phonepe': 'PhonePe', 'frd': 'fraud',
    'frad': 'fraud', 'whats': 'what is', 'aadhar': 'Aadhar', 'adhar': 'Aadhar', 'usne': 'he/she',
    'mere': 'mine', 'apna': 'mine', 'unhone': 'they', 'unka': 'their', 'sbka': 'everyone', 'koi': 'anyone',
    'idhar': 'here', 'udhar': 'there', 'phir': 'again', 'bole': 'said', 'aise': 'like this', 'waise': 'like that',
    'kyu': 'why', 'hai': 'is', 'nahi': 'no', 'kaise': 'how', 'kya': 'what', 'mein': 'in', 'ho': 'is',
    'raha': 'going', 'hu': 'am', 'hindi': 'hindi', 'english': 'english', 'agar': 'if', 'karna': 'do',
    'kar': 'do', 'ke': 'of', 'liye': 'for', 'mera': 'my', 'tera': 'your', 'unka': 'their', 'woh': 'they',
    'bata': 'tell', 'pata': 'know', 'sath': 'with', 'tak': 'until', 'ab': 'now', 'sab': 'all',
    'jaldi': 'quickly', 'kam': 'work', 'paise': 'money', 'aap': 'you', 'tum': 'you', 'kyunki': 'because',
    'kuch': 'some', 'ja': 'go', 'lo': 'take', 'de': 'give', 'mujhe': 'me', 'bhi': 'also', 'baat': 'talk',
    'matlab': 'meaning', 'galat': 'wrong', 'sahi': 'right', 'chori': 'theft', 'fraud': 'fraud',
    'dhokha': 'cheating', 'saath': 'with', 'ghar': 'home', 'jagah': 'place', 'sudhar': 'improve',
    'karo': 'do', 'baar': 'time', 'kaam': 'job', 'zindagi': 'life', 'band': 'closed', 'chalu': 'open',
    'aata': 'comes', 'gaya': 'gone', 'shuru': 'start', 'khatam': 'end', 'samajh': 'understand',
    'tarah': 'way', 'dost': 'friend', 'roko': 'stop', 'soch': 'think', 'dikkat': 'problem', 'madad': 'help',
    'barabar': 'equal', 'jhoot': 'lie', 'asli': 'real', 'nakli': 'fake', 'baaki': 'remaining',
    'pura': 'complete', 'aadmi': 'man', 'aurat': 'woman', 'bacha': 'child', 'report': 'report',
    'shikayat': 'complaint', 'puri': 'full', 'naya': 'new', 'purana': 'old', 'chal': 'move',
    'ruk': 'stop', 'faida': 'benefit', 'nuksan': 'loss', 'samasya': 'issue', 'bade': 'big',
    'chote': 'small', 'dheere': 'slow', 'yahan': 'here', 'wahan': 'there', 'kal': 'yesterday/tomorrow',
    'aaj': 'today', 'bahut': 'very', 'sabse': 'most', 'kamzor': 'weak', 'majboot': 'strong', 'samay': 'time',
    'muft': 'free', 'sasta': 'cheap', 'mehnga': 'expensive', 'bhool': 'forget', 'yaad': 'remember',
    'saaf': 'clear', 'ganda': 'dirty', 'zinda': 'alive', 'marta': 'dying', 'i': 'me', 'rs': 'rupees',
    'ki': 'of', 'se': 'from', 'upi': 'UPI', 'sbi': 'SBI', 'mere': 'mine', 'gariahat': 'Gariahat',
    'ka': 'of', 'india': 'India', 'dont': 'do not', 'pe': 'on', 'aur': 'and', 'kiya': 'done',
    'ac': 'account', 'suchna': 'inform', 'phonepe': 'PhonePe', 'froud': 'fraud', 'kyc': 'KYC',
    'whats': 'what is', 'aadhar': 'Aadhar', 'unhone': 'they'
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


