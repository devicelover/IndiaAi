import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from combined_transformer import CombinedTransformer
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
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
        test_df = test_df[~test_df['category'].isin(unseen_categories)]
        print(f"Removed {len(unseen_categories)} unseen categories from test data.")

    if unseen_sub_categories:
        print(f"Unseen sub-categories in test data: {unseen_sub_categories}")
        test_df = test_df[~test_df['sub_category'].isin(unseen_sub_categories)]
        print(f"Removed {len(unseen_sub_categories)} unseen sub-categories from test data.")

    # Reset index after removals
    test_df.reset_index(drop=True, inplace=True)

    y_train_category = category_encoder.transform(train_df['category'])
    y_train_sub_category = sub_category_encoder.transform(train_df['sub_category'])

    y_test_category = category_encoder.transform(test_df['category'])
    y_test_sub_category = sub_category_encoder.transform(test_df['sub_category'])

    X_train = train_df['crimeaditionalinfo'].fillna('')
    X_test = test_df['crimeaditionalinfo'].fillna('')

    # Transform the data
    print("Transforming the data...")
    transformer = CombinedTransformer()
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    # Initialize models
    category_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        n_jobs=-1
    )
    sub_category_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        n_jobs=-1
    )

    # Training progress bars
    print("Training the models...")
    category_pbar = tqdm(total=category_model.n_estimators, desc="Training Category Model")
    sub_category_pbar = tqdm(total=sub_category_model.n_estimators, desc="Training Sub-Category Model")

    # Custom callback to update progress bar
    def category_callback(env):
        category_pbar.update(1)

    def sub_category_callback(env):
        sub_category_pbar.update(1)

    # Train category model with progress bar
    category_model.fit(
        X_train_transformed,
        y_train_category,
        eval_set=[(X_test_transformed, y_test_category)],
        eval_metric='multi_logloss',
        callbacks=[category_callback]
    )
    category_pbar.close()

    # Train sub-category model with progress bar
    sub_category_model.fit(
        X_train_transformed,
        y_train_sub_category,
        eval_set=[(X_test_transformed, y_test_sub_category)],
        eval_metric='multi_logloss',
        callbacks=[sub_category_callback]
    )
    sub_category_pbar.close()

    # Save the models and transformer
    print("Saving the models and encoders...")
    joblib.dump({
        'transformer': transformer,
        'category_model': category_model,
        'sub_category_model': sub_category_model
    }, 'models/crime_classification_model.joblib')
    joblib.dump(category_encoder, 'models/category_encoder.joblib')
    joblib.dump(sub_category_encoder, 'models/sub_category_encoder.joblib')

    # Evaluate the models
    print("Evaluating the models...")

    # Predict categories
    y_pred_category = category_model.predict(X_test_transformed)
    y_pred_sub_category = sub_category_model.predict(X_test_transformed)

    # Decode labels
    y_test_decoded = pd.DataFrame({
        'category': category_encoder.inverse_transform(y_test_category),
        'sub_category': sub_category_encoder.inverse_transform(y_test_sub_category)
    })
    y_pred_decoded = pd.DataFrame({
        'category': category_encoder.inverse_transform(y_pred_category),
        'sub_category': sub_category_encoder.inverse_transform(y_pred_sub_category)
    })

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
