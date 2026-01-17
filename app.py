from flask import Flask, render_template, request  # Flask for web framework
import pandas as pd  # For handling input data
import joblib  # For loading saved models and scaler
import os  # For checking file paths

# Initialize Flask app
app = Flask(__name__)

# Load Trained Models and Scaler
model_path = os.path.join('model', 'cancer_model.pkl')  # Neural Network
rf_model_path = os.path.join('model', 'random_forest_model.pkl')  # Random Forest
scaler_path = os.path.join('model', 'scaler.pkl')  # Scaler

# Load models and scaler if files exist
model = joblib.load(model_path) if os.path.exists(model_path) else None
rf_model = joblib.load(rf_model_path) if os.path.exists(rf_model_path) else None
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# ===============================
# Route: Home (Prediction Page)
# ===============================
@app.route('/', methods=['GET', 'POST'])
def index():
    models = ["Neural Network", "Random Forest"]  # Dropdown model options
    diagnosis = None  # Prediction result
    error = None  # Error message
    selected_model = "Neural Network"  # Default selected model

    if request.method == 'POST':
        try:
            # Features expected from the form
            feature_names = ['Clump', 'UnifSize', 'UnifShape', 'MargAdh',
                             'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']
            
            # Convert all input values to float
            input_data = [float(request.form[f]) for f in feature_names]

            # Dropdown-selected model
            selected_model = request.form.get("model_type")

            # Additional features used during training
            clump = input_data[0]
            unif_size = input_data[1]
            unif_shape = input_data[2]
            norm_nucl = input_data[7]

            # Prevent division errors
            cell_uniformity_ratio = unif_size / unif_shape if unif_shape != 0 else 0
            nuclei_density = norm_nucl / clump if clump != 0 else 0

            # Append the two new features
            input_data.extend([cell_uniformity_ratio, nuclei_density])

            # Prepare DataFrame for scaling
            input_df = pd.DataFrame([input_data], columns=feature_names + ["CellUniformityRatio", "NucleiDensity"])
            input_scaled = scaler.transform(input_df)

            # Perform prediction
            if selected_model == "Random Forest":
                prediction = rf_model.predict(input_scaled)[0]
            else:
                prediction = model.predict(input_scaled)[0]

            diagnosis = "Malignant" if prediction == 1 else "Benign"

        except Exception as e:
            error = f"Error in input data: {e}"

    return render_template('index.html',
                           diagnosis=diagnosis,
                           error=error,
                           models=models,
                           selected_model=selected_model)

# ===============================
# Route: Visualization Page
# ===============================
@app.route('/visualizations')
def visualizations():
    # Try to read metrics.txt if exists
    metrics_path = os.path.join('static', 'metrics.txt')
    evaluation_text = "No metrics available."

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as file:
                evaluation_text = file.read()
        except:
            evaluation_text = "⚠️ Failed to load metrics."

    return render_template('visualization.html', evaluation_text=evaluation_text)

# ===============================
# Route: About Page
# ===============================
@app.route('/about')
def about():
    return render_template('about.html')

# ===============================
# Start the Flask server
# ===============================
if __name__ == '__main__':
    app.run(debug=True)
