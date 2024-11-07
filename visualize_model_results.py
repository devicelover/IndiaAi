# visualize_model_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

# Set Seaborn style
sns.set(style='whitegrid', palette='muted')

def main():
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('data/test.csv')

    # Standardize labels
    print("Standardizing labels...")
    test_df['category'] = test_df['category'].str.strip().str.lower()
    test_df['sub_category'] = test_df['sub_category'].str.strip().str.lower()
    test_df.dropna(subset=['category', 'sub_category', 'crimeaditionalinfo'], inplace=True)

    # Load encoders and model
    print("Loading encoders and model...")
    category_encoder = joblib.load('models/category_encoder.joblib')
    sub_category_encoder = joblib.load('models/sub_category_encoder.joblib')
    pipeline = joblib.load('models/crime_classification_model.joblib')

    # Handle unseen labels in test data
    print("Handling unseen labels in test data...")
    unseen_categories = set(test_df['category']) - set(category_encoder.classes_)
    unseen_sub_categories = set(test_df['sub_category']) - set(sub_category_encoder.classes_)

    if unseen_categories:
        print(f"Unseen categories in test data: {unseen_categories}")
        test_df = test_df[~test_df['category'].isin(unseen_categories)]

    if unseen_sub_categories:
        print(f"Unseen sub-categories in test data: {unseen_sub_categories}")
        test_df = test_df[~test_df['sub_category'].isin(unseen_sub_categories)]

    test_df.reset_index(drop=True, inplace=True)

    # Encode labels
    print("Encoding labels...")
    y_test = pd.DataFrame({
        'category': category_encoder.transform(test_df['category']),
        'sub_category': sub_category_encoder.transform(test_df['sub_category'])
    })

    X_test = test_df['crimeaditionalinfo'].fillna('')

    # Predict using the loaded model
    print("Predicting on test data...")
    y_pred = pipeline.predict(X_test)

    # Convert predictions to DataFrame
    y_pred_df = pd.DataFrame(y_pred, columns=['category', 'sub_category'])

    # Decode labels
    print("Decoding labels...")
    y_test_decoded = y_test.copy()
    y_pred_decoded = y_pred_df.copy()

    y_test_decoded['category'] = category_encoder.inverse_transform(y_test['category'])
    y_pred_decoded['category'] = category_encoder.inverse_transform(y_pred_df['category'])

    y_test_decoded['sub_category'] = sub_category_encoder.inverse_transform(y_test['sub_category'])
    y_pred_decoded['sub_category'] = sub_category_encoder.inverse_transform(y_pred_df['sub_category'])

    # Generate visualizations
    print("Generating visualizations...")
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Class Distribution Plots
    plot_class_distribution(test_df, 'category', 'Category Distribution in Test Set', output_dir)
    plot_class_distribution(test_df, 'sub_category', 'Sub-category Distribution in Test Set', output_dir)

    # 2. Performance Metrics Plots
    plot_classification_metrics(y_test_decoded['category'], y_pred_decoded['category'],
                                category_encoder.classes_, 'Categories', output_dir)
    plot_classification_metrics(y_test_decoded['sub_category'], y_pred_decoded['sub_category'],
                                sub_category_encoder.classes_, 'Sub-categories', output_dir)

    # 3. Confusion Matrices
    plot_confusion_matrix_cm(y_test_decoded['category'], y_pred_decoded['category'],
                             category_encoder.classes_, 'Categories', output_dir)

    # For sub-categories, plot top N classes to make the matrix readable
    top_n = 10
    plot_top_n_confusion_matrix(y_test_decoded['sub_category'], y_pred_decoded['sub_category'],
                                top_n, 'Sub-categories', output_dir)

    print("Visualizations saved in the 'visualizations' directory.")

def plot_class_distribution(data, column, title, output_dir):
    counts = data[column].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(title)
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{column}_distribution.png')
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_classification_metrics(y_true, y_pred, labels, level, output_dir):
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    metrics = ['precision', 'recall', 'f1-score']
    report_df = report_df[metrics]
    report_df.reset_index(inplace=True)
    report_df.rename(columns={'index': 'Class'}, inplace=True)
    report_df = pd.melt(report_df, id_vars='Class', value_vars=metrics,
                        var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Class', y='Score', hue='Metric', data=report_df)
    plt.title(f'Performance Metrics for {level}')
    plt.xlabel(f'{level}')
    plt.ylabel('Score')
    plt.xticks(rotation=90)
    plt.legend(loc='best')
    plt.tight_layout()
    filename = os.path.join(output_dir, f'performance_metrics_{level.lower()}.png')
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_confusion_matrix_cm(y_true, y_pred, labels, level, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f'Confusion Matrix for {level}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'confusion_matrix_{level.lower()}.png')
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_top_n_confusion_matrix(y_true, y_pred, top_n, level, output_dir):
    top_classes = y_true.value_counts().nlargest(top_n).index.tolist()

    y_true_top = y_true[y_true.isin(top_classes)]
    y_pred_top = y_pred[y_true.isin(top_classes)]

    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=top_classes, yticklabels=top_classes, cmap='Blues')
    plt.title(f'Confusion Matrix for Top {top_n} {level}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'confusion_matrix_top_{top_n}_{level.lower()}.png')
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
