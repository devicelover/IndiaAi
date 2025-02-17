import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from custom_transformers import CombinedTransformer

# Import CombinedTransformer from our custom module
from custom_transformers import CombinedTransformer

warnings.filterwarnings('ignore')

def load_data(train_path, test_path):
    """Load CSV files, clean text columns, and drop rows with missing critical fields."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for col in ['category', 'sub_category']:
        train_df[col] = train_df[col].str.strip().str.lower()
        test_df[col] = test_df[col].str.strip().str.lower()

    train_df = train_df.dropna(subset=['category', 'sub_category', 'crimeaditionalinfo'])
    test_df = test_df.dropna(subset=['category', 'sub_category', 'crimeaditionalinfo'])
    return train_df, test_df

def main():
    start_time = time.time()
    print("Loading data...")
    train_df, test_df = load_data('data/train.csv', 'data/test.csv')

    # Build LabelEncoders and filter test data.
    cat_encoder = LabelEncoder().fit(train_df['category'])
    subcat_encoder = LabelEncoder().fit(train_df['sub_category'])
    test_df = test_df[test_df['category'].isin(cat_encoder.classes_)]
    test_df = test_df[test_df['sub_category'].isin(subcat_encoder.classes_)]
    test_df.reset_index(drop=True, inplace=True)

    # Encode labels.
    y_train_cat = cat_encoder.transform(train_df['category'])
    y_train_subcat = subcat_encoder.transform(train_df['sub_category'])
    y_test_cat = cat_encoder.transform(test_df['category'])
    y_test_subcat = subcat_encoder.transform(test_df['sub_category'])

    # Prepare text data.
    X_train = train_df['crimeaditionalinfo'].fillna('')
    X_test = test_df['crimeaditionalinfo'].fillna('')

    # Generate embeddings using our custom transformer.
    print("Generating embeddings...")
    transformer = CombinedTransformer(device='cpu', batch_size=8)
    X_train_emb = transformer.transform(X_train)
    X_test_emb = transformer.transform(X_test)

    # Train LightGBM classifiers.
    print("Training Category Model...")
    cat_model = LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight='balanced', n_jobs=-1)
    cat_model.fit(X_train_emb, y_train_cat, eval_set=[(X_test_emb, y_test_cat)], eval_metric='multi_logloss')

    print("Training Sub-Category Model...")
    subcat_model = LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight='balanced', n_jobs=-1)
    subcat_model.fit(X_train_emb, y_train_subcat, eval_set=[(X_test_emb, y_test_subcat)], eval_metric='multi_logloss')

    # Evaluate models.
    print("Evaluating models...")
    y_pred_cat = cat_model.predict(X_test_emb)
    y_pred_subcat = subcat_model.predict(X_test_emb)
    print("\nCategory Classification Report:")
    print(classification_report(cat_encoder.inverse_transform(y_test_cat),
                                cat_encoder.inverse_transform(y_pred_cat)))
    print("\nSub-Category Classification Report:")
    print(classification_report(subcat_encoder.inverse_transform(y_test_subcat),
                                subcat_encoder.inverse_transform(y_pred_subcat)))

    # Ensure the 'models' directory exists.
    os.makedirs('models', exist_ok=True)

    # Save models and encoders.
    joblib.dump({
        'transformer': transformer,
        'cat_model': cat_model,
        'subcat_model': subcat_model
    }, 'models/optimized_crime_classification_model.joblib')
    joblib.dump(cat_encoder, 'models/category_encoder.joblib')
    joblib.dump(subcat_encoder, 'models/sub_category_encoder.joblib')

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.2f} minutes.")

if __name__ == '__main__':
    main()
