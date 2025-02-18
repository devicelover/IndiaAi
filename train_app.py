import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
import os

def load_and_preprocess_data(file_path):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["category", "sub_category", "crime_description"], inplace=True)
    df["crime_description"] = df["crime_description"].str.lower().str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    return df

def encode_labels(df):
    """Encodes category and sub-category labels."""
    cat_encoder = LabelEncoder()
    subcat_encoder = LabelEncoder()
    df["category"] = cat_encoder.fit_transform(df["category"])
    df["sub_category"] = subcat_encoder.fit_transform(df["sub_category"])

    # Save encoders
    joblib.dump(cat_encoder, "models/category_encoder.joblib")
    joblib.dump(subcat_encoder, "models/sub_category_encoder.joblib")
    return df, cat_encoder, subcat_encoder

def get_text_embeddings(text_list):
    """Generates sentence embeddings using a pretrained model."""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(text_list, convert_to_tensor=True)
    return embeddings.cpu().detach().numpy()

def train_and_save_model(X_train, X_test, y_train, y_test, model_name):
    """Trains a LightGBM classifier and saves the model."""
    model = LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="multi_logloss")
    joblib.dump(model, f"models/{model_name}.joblib")
    return model

def main():
    os.makedirs("models", exist_ok=True)

    print("Loading and preprocessing data...")
    data_path = "data/cleaned_crime_data.csv"
    df = load_and_preprocess_data(data_path)
    df, cat_encoder, subcat_encoder = encode_labels(df)

    print("Generating embeddings...")
    X_embeddings = get_text_embeddings(df["crime_description"].tolist())
    y_category = df["category"].values
    y_sub_category = df["sub_category"].values

    # Train-test split
    X_train, X_test, y_train_cat, y_test_cat = train_test_split(X_embeddings, y_category, test_size=0.2, random_state=42)
    X_train, X_test, y_train_subcat, y_test_subcat = train_test_split(X_embeddings, y_sub_category, test_size=0.2, random_state=42)

    print("Training category model...")
    cat_model = train_and_save_model(X_train, X_test, y_train_cat, y_test_cat, "optimized_crime_classification_model")

    print("Training sub-category model...")
    subcat_model = train_and_save_model(X_train, X_test, y_train_subcat, y_test_subcat, "sub_category_model")

    print("Evaluating models...")
    y_pred_cat = cat_model.predict(X_test)
    y_pred_subcat = subcat_model.predict(X_test)

    print("\nCategory Classification Report:")
    print(classification_report(cat_encoder.inverse_transform(y_test_cat), cat_encoder.inverse_transform(y_pred_cat)))

    print("\nSub-Category Classification Report:")
    print(classification_report(subcat_encoder.inverse_transform(y_test_subcat), subcat_encoder.inverse_transform(y_pred_subcat)))

    print("Model training completed and saved!")

if __name__ == "__main__":
    main()
