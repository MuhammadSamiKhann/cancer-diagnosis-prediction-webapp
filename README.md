# 🧠 Cancer Diagnosis Prediction System

## 📌 Introduction
This project focuses on predicting whether a tumor is **benign (non-cancerous)** or **malignant (cancerous)** using machine learning techniques.  
The goal was to build a **complete, end-to-end system** starting from a real-world dataset, applying proper preprocessing, training multiple models, comparing their performance, and deploying the solution through a **Flask-based web interface**.

### Project Objectives
- Use real cancer diagnosis data
- Clean and preprocess the dataset
- Train at least two machine learning models
- Compare model performance
- Deploy the models using a Flask web application
- Provide visual analysis and graphs for better understanding

---

## 📊 Dataset Description
The project uses the **Wisconsin Breast Cancer Diagnostic Dataset**, which contains **699 tumor samples** with medical measurements.

### Features Included
- Clump Thickness  
- Uniformity of Cell Size  
- Uniformity of Cell Shape  
- Marginal Adhesion  
- Single Epithelial Cell Size  
- Bare Nuclei  
- Bland Chromatin  
- Normal Nucleoli  
- Mitoses  

### Target Variable
- **Class**
  - `2 → Benign`
  - `4 → Malignant`

The target variable was converted to binary values:
- `0 = Benign`
- `1 = Malignant`

---

## 🧹 Data Preprocessing

### Handling Missing Values
- The `BareNuc` column contained missing and non-numeric values
- These were converted using pandas
- Rows with missing values were removed to avoid prediction errors

### Data Formatting
- All feature columns were converted to numeric types
- The class label was mapped from `2/4` to `0/1`

### Outlier Detection
- Outliers were detected and removed using the **Z-score method**
- Rows with Z-scores greater than **3** were excluded

### Normalization
- Feature scaling was applied using a standard scaler to improve model performance

---

## 🧠 Feature Engineering
Two new features were created to enhance model learning:

- **CellUniformityRatio**  
  `Uniformity of Cell Size / Uniformity of Cell Shape`

- **NucleiDensity**  
  `Normal Nucleoli / Clump Thickness`

These engineered features helped capture relationships between important cell properties.

---

## 🤖 Model Development

### Models Used
1. **Neural Network (MLPClassifier)**  
   - Multi-layer perceptron with hidden layers  
   - Captures complex non-linear patterns  

2. **Random Forest Classifier**  
   - Ensemble model using multiple decision trees  
   - Provides feature importance analysis  

### Training Strategy
- **80–20 train-test split**
- Stratified sampling to preserve class balance

---

## 📈 Model Evaluation

### Evaluation Metrics
- Accuracy
- Confusion Matrix
- Training Loss Curve (Neural Network)
- Feature Importance (Random Forest)

### Model Saving
- Both trained models and the scaler were saved using **joblib**
- This allows easy reuse inside the Flask application

---

## 📊 Visualizations & Analysis

### Graphs Included
- Confusion matrix for both models
- Neural network training loss curve
- Random forest feature importance plot
- Histograms for all features
- Correlation heatmap
- Boxplots comparing benign vs malignant cases

### Key Observations
- Malignant tumors usually show higher values for:
  - Bare Nuclei
  - Uniformity of Cell Size
  - Uniformity of Cell Shape
- Random Forest clearly highlighted the most influential features
- Neural Network achieved slightly higher accuracy but required careful preprocessing

---
Data Collection
↓
Preprocessing
↓
Feature Engineering
↓
Model Training
↓
Evaluation
↓
Visualization
↓
Web Deployment


All steps were implemented using **Python**, **scikit-learn**, **pandas**, and **Flask**.

---

## 🌐 Web Application

### Features
- Simple and user-friendly interface
- Input form for medical measurements
- Model selection:
  - Neural Network
  - Random Forest
- Displays prediction result instantly

### Visualization Pages
- Confusion matrix
- Neural network loss curve
- Feature importance chart
- Correlation heatmap

---

## 🧾 Conclusion
This project successfully achieved its objectives:
- Cleaned and prepared a real-world cancer dataset
- Trained and compared two classification models
- Built a functional Flask web application
- Used visual analysis to understand model performance

This project demonstrates the **complete machine learning pipeline**, from data preprocessing to deployment.

---

## 🛠️ Technologies Used
- Python  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  
- Flask  
- Joblib  

---

## 🔄 Workflow
