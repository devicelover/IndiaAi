# train_model.py

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from combined_transformer import CombinedTransformer
from sklearn.metrics import classification_report
from tqdm import tqdm
import time

def main():
    start_time = time.time()

    # Load data
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    # Standardize labels
    print("Standardizing labels...")
    for df in [train_df, test_df]:
        df['category'] = df['category'].str.strip().str.lower()
        df['sub_category'] = df['sub_category'].str.strip().str.lower()

    # Handle missing values
    print("Handling missing values...")
    train_df.dropna(subset=['category', 'sub_category', 'crimeaditionalinfo'], inplace=True)
    test_df.dropna(subset=['category', 'sub_category', 'crimeaditionalinfo'], inplace=True)

    # Encode labels
    print("Encoding labels...")
    category_encoder = LabelEncoder()
    sub_category_encoder = LabelEncoder()

    category_encoder.fit(train_df['category'])
    sub_category_encoder.fit(train_df['sub_category'])

    # Identify unseen categories and sub-categories in test data
    unseen_categories = set(test_df['category']) - set(category_encoder.classes_)
    unseen_sub_categories = set(test_df['sub_category']) - set(sub_category_encoder.classes_)

    if unseen_categories:
        print(f"Unseen categories in test data: {unseen_categories}")
        # Remove rows with unseen categories
        test_df = test_df[~test_df['category'].isin(unseen_categories)]
        print(f"Removed {len(unseen_categories)} unseen categories from test data.")

    if unseen_sub_categories:
        print(f"Unseen sub-categories in test data: {unseen_sub_categories}")
        # Remove rows with unseen sub-categories
        test_df = test_df[~test_df['sub_category'].isin(unseen_sub_categories)]
        print(f"Removed {len(unseen_sub_categories)} unseen sub-categories from test data.")

    # Reset index after removals
    test_df.reset_index(drop=True, inplace=True)

    y_train = pd.DataFrame({
        'category': category_encoder.transform(train_df['category']),
        'sub_category': sub_category_encoder.transform(train_df['sub_category'])
    })

    y_test = pd.DataFrame({
        'category': category_encoder.transform(test_df['category']),
        'sub_category': sub_category_encoder.transform(test_df['sub_category'])
    })

    X_train = train_df['crimeaditionalinfo'].fillna('')
    X_test = test_df['crimeaditionalinfo'].fillna('')

    # Build the pipeline
    print("Building the pipeline...")
    pipeline = Pipeline([
        ('combined_transformer', CombinedTransformer()),
        ('classifier', MultiOutputClassifier(LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            n_jobs=-1
        )))
    ])

    # Train the model with progress bar
    print("Training the model...")
    with tqdm(total=100, desc="Training Progress") as pbar:
        pipeline.fit(X_train, y_train)
        pbar.update(100)

    # Save the model and encoders
    print("Saving the model and encoders...")
    joblib.dump(pipeline, 'models/crime_classification_model.joblib')
    joblib.dump(category_encoder, 'models/category_encoder.joblib')
    joblib.dump(sub_category_encoder, 'models/sub_category_encoder.joblib')

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)

    # Convert predictions to DataFrame
    y_pred_df = pd.DataFrame(y_pred, columns=['category', 'sub_category'])

    # Decode labels
    y_test_decoded = y_test.copy()
    y_pred_decoded = y_pred_df.copy()

    y_test_decoded['category'] = category_encoder.inverse_transform(y_test['category'])
    y_pred_decoded['category'] = category_encoder.inverse_transform(y_pred_df['category'])

    y_test_decoded['sub_category'] = sub_category_encoder.inverse_transform(y_test['sub_category'])
    y_pred_decoded['sub_category'] = sub_category_encoder.inverse_transform(y_pred_df['sub_category'])

    # Classification reports
    print("\nCategory Classification Report:")
    print(classification_report(y_test_decoded['category'], y_pred_decoded['category']))

    print("\nSub-category Classification Report:")
    print(classification_report(y_test_decoded['sub_category'], y_pred_decoded['sub_category']))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nModel training and evaluation completed in {total_time:.2f} seconds.")

if __name__ == '__main__':
    main()
