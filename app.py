# app.py

from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
import os
import time
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

# Import the TextPreprocessor class from text_preprocessor.py
from text_preprocessor import TextPreprocessor

# Initialize NLTK components
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the model and encoders
model = joblib.load('models/crime_classification_model.joblib')
category_encoder = joblib.load('models/category_encoder.joblib')
sub_category_encoder = joblib.load('models/sub_category_encoder.joblib')

# Since the model pipeline includes preprocessing and embedding, we don't need to load the embedding model separately

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check which form was submitted
        if 'text_input' in request.form:
            # Text input form submitted
            description = request.form['description']
            if description.strip() == '':
                flash('Please enter a description.')
                return redirect(url_for('index'))
            else:
                # Process the input and get prediction
                # Since our model includes preprocessing and embedding, we can directly predict
                prediction = model.predict([description])
                category_pred = category_encoder.inverse_transform(prediction[0])[0]
                sub_category_pred = sub_category_encoder.inverse_transform(prediction[1])[0]
                return render_template('result.html', description=description, category=category_pred, sub_category=sub_category_pred)
        elif 'file_input' in request.form:
            # File upload form submitted
            if 'file' not in request.files or request.files['file'].filename == '':
                flash('No file selected.')
                return redirect(url_for('index'))
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(input_filepath)

                # Process the file
                output_filepath, eta = process_file(input_filepath)

                # Provide the download link and ETA
                return render_template('download.html', output_filename=os.path.basename(output_filepath), eta=eta)
            else:
                flash('Invalid file type. Please upload a CSV file.')
                return redirect(url_for('index'))
    return render_template('index.html')

def process_file(filepath):
    start_time = time.time()
    df = pd.read_csv(filepath)

    if 'crimeaditionalinfo' not in df.columns:
        flash('CSV file must contain a "crimeaditionalinfo" column.')
        return redirect(url_for('index'))

    # Since our model pipeline includes preprocessing and embedding, we can directly predict
    # Predict
    predictions = model.predict(df['crimeaditionalinfo'])

    # Decode predictions
    categories = category_encoder.inverse_transform(predictions[0])
    sub_categories = sub_category_encoder.inverse_transform(predictions[1])

    # Add predictions to DataFrame
    df['Predicted Category'] = categories
    df['Predicted Sub-category'] = sub_categories

    # Save the processed file
    output_filename = os.path.basename(filepath).rsplit('.', 1)[0] + '_processed.csv'
    output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    df.to_csv(output_filepath, index=False)

    end_time = time.time()
    processing_time = end_time - start_time
    eta = f"Processing completed in {processing_time:.2f} seconds."

    flash(eta)
    return output_filepath, eta

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
