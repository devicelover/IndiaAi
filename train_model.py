# train_model.py

import pandas as pd
import numpy as np
import nltk
import warnings
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from text_preprocessor import TextPreprocessor

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize NLTK components
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define the EmbeddingTransformer class
class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a list of strings
        if isinstance(X, (pd.Series, pd.DataFrame)):
            texts = X.tolist()
        elif isinstance(X, np.ndarray):
            texts = X.tolist()
        elif isinstance(X, list):
            texts = X  # X is already a list
        else:
            texts = [str(X)]
        return self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)

def main():
    # Load the data
    print("Loading data...")
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' and/or 'test.csv' not found in 'data/' directory.")
        return

    # Standardize labels in both datasets
    print("Standardizing labels...")
    for df in [train_df, test_df]:
        df['category'] = df['category'].str.strip().str.lower()
        df['sub_category'] = df['sub_category'].str.strip().str.lower()

    # Drop rows with NaN in 'category' or 'sub_category'
    print("Dropping rows with NaN values in 'category' or 'sub_category'...")
    train_df.dropna(subset=['category', 'sub_category'], inplace=True)
    test_df.dropna(subset=['category', 'sub_category'], inplace=True)

    # Reset index after dropping rows
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Prepare the labels
    print("Encoding labels...")
    category_encoder = LabelEncoder()
    sub_category_encoder = LabelEncoder()

    # Fit encoders on training data
    category_encoder.fit(train_df['category'])
    sub_category_encoder.fit(train_df['sub_category'])

    # Remove unseen categories and sub-categories from test data
    print("Removing unseen categories and sub-categories from test data...")
    test_df = test_df[test_df['category'].isin(category_encoder.classes_)]
    test_df = test_df[test_df['sub_category'].isin(sub_category_encoder.classes_)]
    test_df.reset_index(drop=True, inplace=True)

    # Encode labels
    y_train = pd.DataFrame({
        'category': category_encoder.transform(train_df['category']),
        'sub_category': sub_category_encoder.transform(train_df['sub_category'])
    })

    y_test = pd.DataFrame({
        'category': category_encoder.transform(test_df['category']),
        'sub_category': sub_category_encoder.transform(test_df['sub_category'])
    })

    # Split the data
    X_train = train_df['crimeaditionalinfo'].fillna('')
    X_test = test_df['crimeaditionalinfo'].fillna('')

    # Build the pipeline with the custom EmbeddingTransformer
    print("Building pipeline...")
    pipeline = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('embedding', EmbeddingTransformer()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            class_weight='balanced',
            random_state=42
        )))
    ])

    # Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Save the model for future use
    print("Saving model...")
    joblib.dump(pipeline, 'models/crime_classification_model.joblib')
    joblib.dump(category_encoder, 'models/category_encoder.joblib')
    joblib.dump(sub_category_encoder, 'models/sub_category_encoder.joblib')

    # Evaluate the model
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)

    # Convert predictions to DataFrame
    y_pred_df = pd.DataFrame(y_pred, columns=['category', 'sub_category'])

    # Decode the labels
    y_test_decoded = y_test.copy()
    y_pred_decoded = y_pred_df.copy()

    y_test_decoded['category'] = category_encoder.inverse_transform(y_test['category'])
    y_pred_decoded['category'] = category_encoder.inverse_transform(y_pred_df['category'])

    y_test_decoded['sub_category'] = sub_category_encoder.inverse_transform(y_test['sub_category'])
    y_pred_decoded['sub_category'] = sub_category_encoder.inverse_transform(y_pred_df['sub_category'])

    # Classification report for category
    print("\nCategory Classification Report:")
    print(classification_report(y_test_decoded['category'], y_pred_decoded['category'], zero_division=0))

    # Classification report for sub_category
    print("\nSub-category Classification Report:")
    print(classification_report(y_test_decoded['sub_category'], y_pred_decoded['sub_category'], zero_division=0))

    print("\nModel training and evaluation completed.")

if __name__ == '__main__':
    main()
