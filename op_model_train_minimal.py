import os
# Set these environment variables early.
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import warnings
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

# Import torch and set number of threads explicitly
import torch
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')


# -------------------------------
# CombinedTransformer (Minimal Version)
# -------------------------------
class CombinedTransformer:
    """
    A minimal transformer that cleans text using a small normalization dictionary
    and generates embeddings using SentenceTransformer on CPU.
    """
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', batch_size=1, device='cpu'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device  # Forced CPU for stability
        self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
        # A small normalization dictionary – extend as needed.
        self.normalization_dict = {
            'kyu': 'why',
            'hai': 'is',
            'nahi': 'no',
            'kaise': 'how',
            'kya': 'what'
        }

    def clean_texts(self, texts):
        """Lowercase, remove non-alphanumeric characters, and apply normalization."""
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


# -------------------------------
# Minimal Test for Embedding Generation
# -------------------------------
def test_embedding_generation():
    print("Running a quick embedding test...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    texts = ["This is a test.", "Another example sentence."]
    embeddings = model.encode(texts, batch_size=1, show_progress_bar=True)
    print("Test embeddings shape:", np.array(embeddings).shape)


# -------------------------------
# Data Loading Function (Small Subset)
# -------------------------------
def load_data_small(train_path, test_path, nrows=50):
    """
    Load a small subset of train and test CSV files using pandas,
    clean text columns, and drop rows with missing critical fields.
    """
    train_df = pd.read_csv(train_path, nrows=nrows)
    test_df = pd.read_csv(test_path, nrows=nrows)

    for col in ['category', 'sub_category']:
        train_df[col] = train_df[col].str.strip().str.lower()
        test_df[col] = test_df[col].str.strip().str.lower()

    train_df = train_df.dropna(subset=['category', 'sub_category', 'crimeaditionalinfo'])
    test_df = test_df.dropna(subset=['category', 'sub_category', 'crimeaditionalinfo'])

    return train_df, test_df


# -------------------------------
# Main Pipeline (Minimal)
# -------------------------------
def main():
    start_time = time.time()

    # Run a quick test for embedding generation
    test_embedding_generation()

    print("Loading a small subset of data...")
    train_df, test_df = load_data_small("data/train.csv", "data/test.csv", nrows=50)

    # Use LabelEncoder for target columns.
    cat_encoder = LabelEncoder().fit(train_df["category"])
    subcat_encoder = LabelEncoder().fit(train_df["sub_category"])

    # Filter test data to only include known classes.
    test_df = test_df[test_df["category"].isin(cat_encoder.classes_)]
    test_df = test_df[test_df["sub_category"].isin(subcat_encoder.classes_)]
    test_df.reset_index(drop=True, inplace=True)

    # Encode labels.
    y_train_cat = cat_encoder.transform(train_df["category"])
    y_train_subcat = subcat_encoder.transform(train_df["sub_category"])
    y_test_cat = cat_encoder.transform(test_df["category"])
    y_test_subcat = subcat_encoder.transform(test_df["sub_category"])

    # Prepare text data.
    X_train = train_df["crimeaditionalinfo"].fillna("")
    X_test = test_df["crimeaditionalinfo"].fillna("")

    print("Generating embeddings (this may take a while)...")
    transformer = CombinedTransformer(device="cpu", batch_size=1)
    X_train_emb = transformer.transform(X_train)
    X_test_emb = transformer.transform(X_test)

    print("Training Category Model...")
    cat_model = LGBMClassifier(n_estimators=10, learning_rate=0.05, class_weight="balanced", n_jobs=-1)
    cat_model.fit(X_train_emb, y_train_cat)

    print("Training Sub-Category Model...")
    subcat_model = LGBMClassifier(n_estimators=10, learning_rate=0.05, class_weight="balanced", n_jobs=-1)
    subcat_model.fit(X_train_emb, y_train_subcat)

    print("Evaluating models...")
    y_pred_cat = cat_model.predict(X_test_emb)
    y_pred_subcat = subcat_model.predict(X_test_emb)
    print("\nCategory Classification Report:")
    print(classification_report(cat_encoder.inverse_transform(y_test_cat),
                                cat_encoder.inverse_transform(y_pred_cat)))
    print("\nSub-Category Classification Report:")
    print(classification_report(subcat_encoder.inverse_transform(y_test_subcat),
                                subcat_encoder.inverse_transform(y_pred_subcat)))

    # Save models and encoders.
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        'transformer': transformer,
        'cat_model': cat_model,
        'subcat_model': subcat_model
    }, "models/optimized_crime_classification_model.joblib")
    joblib.dump(cat_encoder, "models/category_encoder.joblib")
    joblib.dump(subcat_encoder, "models/sub_category_encoder.joblib")

    elapsed = time.time() - start_time
    print(f"Training and evaluation completed in {elapsed/60:.2f} minutes.")

if __name__ == "__main__":
    main()
parents recivid a whatsapp call where fraudiyas told them aapka beta ko CBI ne giraftaar kiya hai. agar tamne riha krwanu che tih 50 hazaar amne transfer karo. they had my all personal details i don’t know how
