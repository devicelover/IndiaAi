from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename
from custom_transformers import CombinedTransformer


app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a strong secret key

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

print("Loading new model and encoders...")
models = joblib.load('models/optimized_crime_classification_model.joblib')
transformer = models['transformer']
cat_model = models['cat_model']
subcat_model = models['subcat_model']
cat_encoder = joblib.load('models/category_encoder.joblib')
subcat_encoder = joblib.load('models/sub_category_encoder.joblib')

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.context_processor
def inject_current_year():
    from datetime import datetime
    return {'current_year': datetime.now().year}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'text_input' in request.form:
            description = request.form.get('description', '').strip()
            if description == '':
                flash('Please enter a description.')
                return redirect(url_for('index'))
            else:
                X_transformed = transformer.transform([description])
                predicted_cat_index = cat_model.predict(X_transformed)[0]
                predicted_subcat_index = subcat_model.predict(X_transformed)[0]
                try:
                    category_pred = cat_encoder.inverse_transform([predicted_cat_index])[0]
                except Exception as e:
                    print("Error in inverse transforming category:", e)
                    category_pred = "Unknown Category"
                try:
                    sub_category_pred = subcat_encoder.inverse_transform([predicted_subcat_index])[0]
                except Exception as e:
                    print("Error in inverse transforming sub-category:", e)
                    sub_category_pred = "Unknown Sub-category"
                return render_template('result.html', description=description,
                                       category=category_pred, sub_category=sub_category_pred)
        elif 'file_input' in request.form:
            file = request.files.get('file')
            if not file or file.filename == '':
                flash('No file selected.')
                return redirect(url_for('index'))
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(input_filepath)
                output_filepath = process_file(input_filepath)
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
    X_transformed = transformer.transform(df['crimeaditionalinfo'].fillna(''))
    predicted_cat_indices = cat_model.predict(X_transformed)
    predicted_subcat_indices = subcat_model.predict(X_transformed)
    categories = cat_encoder.inverse_transform(predicted_cat_indices)
    sub_categories = subcat_encoder.inverse_transform(predicted_subcat_indices)
    df['Predicted Category'] = categories
    df['Predicted Sub-category'] = sub_categories
    output_filename = os.path.basename(filepath).rsplit('.', 1)[0] + '_processed.csv'
    output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    df.to_csv(output_filepath, index=False)
    return output_filepath

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

@app.route('/test')
def test():
    return "Flask is running and responding!"

if __name__ == '__main__':
    app.run(debug=True)
