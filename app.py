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

# Load the embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Function to clean text (same as in your model training script)
def clean_text(text):
    # Handle non-string inputs
    if not isinstance(text, str):
        text = '' if pd.isnull(text) else str(text)

    # Lowercase and remove punctuation
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())

    # Normalization dictionary
    normalization_dict = {
        # ... (Include your normalization terms here)
        'kyu': 'why',
        'hai': 'is',
        # Add all other terms as needed
    }

    words = text.split()
    words = [normalization_dict.get(w, w) for w in words]
    return ' '.join(words)

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
                cleaned_text = clean_text(description)
                embedding = embedding_model.encode([cleaned_text])
                prediction = model.predict(embedding)
                category_pred = category_encoder.inverse_transform(prediction[0])[0]
                sub_category_pred = sub_category_encoder.inverse_transform(prediction[1])[0]
                return render_template('result.html', description=description, category=category_pred, sub_category=sub_category_pred)
        elif 'file_input' in request.files:
            # File upload form submitted
            file = request.files['file']
            if file.filename == '':
                flash('No file selected.')
                return redirect(url_for('index'))
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(input_filepath)

                # Process the file in the background
                output_filename = process_file(input_filepath)

                return send_file(output_filename, as_attachment=True)
            else:
                flash('Invalid file type. Please upload a CSV file.')
                return redirect(url_for('index'))
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def process_file(filepath):
    start_time = time.time()
    df = pd.read_csv(filepath)

    if 'crimeaditionalinfo' not in df.columns:
        flash('CSV file must contain a "crimeaditionalinfo" column.')
        return redirect(url_for('index'))

    # Clean and preprocess the text
    df['cleaned_text'] = df['crimeaditionalinfo'].apply(clean_text)

    # Compute embeddings
    embeddings = embedding_model.encode(df['cleaned_text'].tolist(), batch_size=32, show_progress_bar=True)

    # Predict
    predictions = model.predict(embeddings)

    # Decode predictions
    categories = category_encoder.inverse_transform(predictions[0])
    sub_categories = sub_category_encoder.inverse_transform(predictions[1])

    # Add predictions to DataFrame
    df['Predicted Category'] = categories
    df['Predicted Sub-category'] = sub_categories

    # Remove temporary columns
    df.drop(['cleaned_text'], axis=1, inplace=True)

    # Save the processed file
    output_filename = os.path.basename(filepath).rsplit('.', 1)[0] + '_processed.csv'
    output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    df.to_csv(output_filepath, index=False)

    end_time = time.time()
    processing_time = end_time - start_time
    eta = f"Processing completed in {processing_time:.2f} seconds."

    flash(eta)
    return output_filepath


@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
