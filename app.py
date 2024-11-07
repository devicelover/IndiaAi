# app.py

from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with your actual secret key

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the model and encoders
print("Loading model and encoders...")
model = joblib.load('models/crime_classification_model.joblib')
category_encoder = joblib.load('models/category_encoder.joblib')
sub_category_encoder = joblib.load('models/sub_category_encoder.joblib')

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.context_processor
def inject_current_year():
    from datetime import datetime
    return {'current_year': datetime.now().year}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check which form was submitted
        if 'text_input' in request.form:
            # Text input form submitted
            description = request.form.get('description', '').strip()
            if description == '':
                flash('Please enter a description.')
                return redirect(url_for('index'))
            else:
                # Process the input and get prediction
                prediction = model.predict([description])

                # Access the predicted indices
                predicted_category_index = prediction[0][0]
                predicted_sub_category_index = prediction[0][1]

                # Perform inverse transformation
                try:
                    category_pred = category_encoder.inverse_transform([predicted_category_index])[0]
                except ValueError as e:
                    print("Error:", e)
                    category_pred = "Unknown Category"

                try:
                    sub_category_pred = sub_category_encoder.inverse_transform([predicted_sub_category_index])[0]
                except ValueError as e:
                    print("Error:", e)
                    sub_category_pred = "Unknown Sub-category"

                return render_template('result.html', description=description, category=category_pred, sub_category=sub_category_pred)
        elif 'file_input' in request.form:
            # File upload form submitted
            file = request.files.get('file')
            if not file or file.filename == '':
                flash('No file selected.')
                return redirect(url_for('index'))
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(input_filepath)

                # Process the file
                output_filepath = process_file(input_filepath)

                # Provide the download link
                return render_template('download.html', output_filename=os.path.basename(output_filepath))
            else:
                flash('Invalid file type. Please upload a CSV file.')
                return redirect(url_for('index'))
    return render_template('index.html')

def process_file(filepath):
    df = pd.read_csv(filepath)

    if 'crimeaditionalinfo' not in df.columns:
        flash('CSV file must contain a "crimeaditionalinfo" column.')
        return redirect(url_for('index'))

    # Predict using the model pipeline
    predictions = model.predict(df['crimeaditionalinfo'].fillna(''))

    # Extract predicted indices
    predicted_category_indices = predictions[:, 0]
    predicted_sub_category_indices = predictions[:, 1]

    # Decode predictions
    categories = category_encoder.inverse_transform(predicted_category_indices)
    sub_categories = sub_category_encoder.inverse_transform(predicted_sub_category_indices)

    # Add predictions to DataFrame
    df['Predicted Category'] = categories
    df['Predicted Sub-category'] = sub_categories

    # Save the processed file
    output_filename = os.path.basename(filepath).rsplit('.', 1)[0] + '_processed.csv'
    output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    df.to_csv(output_filepath, index=False)

    return output_filepath

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
