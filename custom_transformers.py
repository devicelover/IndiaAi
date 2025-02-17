# custom_transformers.py

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class CombinedTransformer:
    """
    Cleans text using an extensive normalization dictionary and then generates embeddings
    using SentenceTransformer. Designed to support multi-lingual text.
    """
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', batch_size=8, device='cpu'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device  # e.g., 'cpu', 'cuda', or 'mps'
        self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
        # Extend this dictionary as needed.
        self.normalization_dict = {
           'kyu': 'why', 'ky': 'why', 'kyo': 'why', 'kuy': 'why',
            'hai': 'is', 'ha': 'is', 'h': 'is',
            'nhi': 'no', 'nahi': 'no', 'nahin': 'no', 'nah': 'no',
            'kaise': 'how', 'kaisee': 'how', 'kese': 'how', 'kaisa': 'how',
            'kya': 'what', 'kia': 'what', 'ky': 'what', 'kyya': 'what',
            'koe': 'anyone', 'koi': 'anyone',
            'kuch': 'some', 'kch': 'some', 'kucch': 'some',
            'mein': 'in', 'mai': 'in', 'me': 'in',
            'ho': 'is', 'hoon': 'am', 'hu': 'am', 'hun': 'am',
            'hindhi': 'hindi', 'hindi': 'hindi', 'hnd': 'hindi',
            'angrezi': 'english', 'eng': 'english', 'ang': 'english',
            'agar': 'if', 'agr': 'if', 'ager': 'if',
            'karna': 'do', 'krna': 'do', 'kr': 'do', 'kar': 'do', 'kre': 'do',
            'ke': 'of', 'k': 'of',
            'lye': 'for', 'liye': 'for',
            'mera': 'my', 'meri': 'my', 'mra': 'my',
            'ter': 'your', 'tera': 'your', 'tere': 'your', 'tumhara': 'your', 'tumhare': 'your',
            'unka': 'their', 'unke': 'their', 'unk': 'their',
            'wo': 'they', 'woah': 'they', 'woh': 'they',
            'bataya': 'told', 'btaya': 'told',
            'batana': 'tell', 'bta': 'tell', 'bata': 'tell',
            'pata': 'know', 'ptha': 'know',
            'sath': 'with', 'saath': 'with', 'sathh': 'with',
            'takk': 'until', 'tak': 'until',
            'ab': 'now',
            'sab': 'all', 'sbb': 'all',
            'sabhi': 'everyone', 'sbhi': 'everyone',
            'jldi': 'quickly', 'jaldi': 'quickly',
            'km': 'work', 'kam': 'work',
            'kaam': 'job',
            'paisa': 'money', 'pese': 'money',
            'aap': 'you', 'app': 'you', 'ap': 'you',
            'tum': 'you', 'tu': 'you', 'ty': 'you',
            'kyunki': 'because', 'kuki': 'because', 'kyuki': 'because',
            'j': 'go', 'jaa': 'go',
            'lo': 'take',
            'de': 'give', 'dediya': 'given', 'dedo': 'give',
            'mujhe': 'me', 'mjh': 'me', 'mjhe': 'me', 'mujh': 'me',
            'bhi': 'also',
            'bhae': 'brother', 'bhaiya': 'brother',
            'bhayo': 'brothers', 'bhaiyo': 'brothers',
            'baat': 'talk', 'bat': 'talk',
            'matlab': 'meaning', 'mtlb': 'meaning', 'mltb': 'meaning',
            'glat': 'wrong', 'galat': 'wrong', 'galath': 'wrong',
            'phn': 'phone', 'fone': 'phone', 'fon': 'phone',
            'fraud': 'fraud', 'frd': 'fraud',
            'dhoka': 'cheating', 'dhokha': 'cheating',
            'ghar': 'home',
            'jga': 'place',
            'sudar': 'improve', 'sudhar': 'improve',
            'karo': 'do', 'krdo': 'do', 'kardo': 'do',
            'baar': 'time', 'bar': 'time',
            'zindgee': 'life', 'zindagi': 'life',
            'bnd': 'closed', 'band': 'closed',
            'chl': 'open', 'chalu': 'open', 'chlau': 'open',
            'ata': 'comes', 'aata': 'comes', 'aya': 'comes',
            'gya': 'gone',
            'khtam': 'end', 'khatam': 'end',
            'smjh': 'understand', 'samaj': 'understand', 'samajh': 'understand',
            'tarah': 'way',
            'dost': 'friend', 'dosth': 'friend', 'frnd': 'friend', 'frndz': 'friends',
            'soch': 'think',
            'dkt': 'problem',
            'madat': 'help', 'mdad': 'help',
            'jhooth': 'lie', 'jhoot': 'lie',
            'asli': 'real', 'aadi': 'half',
            'aadmi': 'man', 'mrd': 'man',
            'aurat': 'woman', 'orath': 'woman',
            'bacha': 'child',
            'childr': 'children', 'bacho': 'children',
            'shikayat': 'complaint', 'complain': 'complaint',
            'poorna': 'complete',
            'naya': 'new', 'purana': 'old',
            'jao': 'go', 'jayega': 'will go',
            'faida': 'benefit', 'bhai': 'brother',
            'badi': 'big', 'bade': 'big',
            'chota': 'small',
            'kma': 'earn',
            'deere': 'slow', 'dhire': 'slow',
            'yaha': 'here', 'yahan': 'here',
            'waha': 'there',
            'kal': 'yesterday/tomorrow',
            'aaj': 'today',
            'bahut': 'very', 'bhut': 'very',
            'sabse': 'most',
            'kamzor': 'weak', 'majboot': 'strong',
            'samay': 'time', 'smye': 'time',
            'muft': 'free',
            'sasta': 'cheap',
            'mahnga': 'expensive', 'mehnga': 'expensive',
            'bhool': 'forget', 'bhul': 'forget',
            'yad': 'remember', 'yaad': 'remember',
            'saaf': 'clear', 'clear': 'clear',
            'gnda': 'dirty',
            'zinda': 'alive',
            'mt': 'not',
            'otp': 'OTP',
            'suchna': 'inform', 'inform': 'inform',
            'phonepe': 'PhonePe',
            'whats': 'what is',
            'aadhar': 'Aadhar', 'adhar': 'Aadhar',
            'usne': 'he/she',
            'mere': 'mine', 'apna': 'mine',
            'unhone': 'they', 'unka': 'their',
            'sbka': 'everyone',
            'koi': 'anyone',
            'idhar': 'here', 'udhar': 'there',
            'phir': 'again',
            'bole': 'said',
            'aise': 'like this', 'waise': 'like that'
        }

    def clean_texts(self, texts):
        """Convert texts to lowercase, remove non-alphanumeric characters, and normalize words."""
        texts = texts.str.lower().fillna("")
        texts = texts.apply(lambda x: re.sub(r"[^\w\s]", "", x))
        texts = texts.apply(lambda x: " ".join(self.normalization_dict.get(word, word) for word in x.split()))
        return texts

    def transform(self, texts):
        """Clean texts and return embeddings as a NumPy array."""
        if not isinstance(texts, pd.Series):
            texts = pd.Series(texts)
        cleaned_texts = self.clean_texts(texts)
        embeddings = self.embedding_model.encode(
            cleaned_texts.tolist(),
            batch_size=self.batch_size,
            show_progress_bar=True
        )
        return np.array(embeddings)
