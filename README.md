# 🧠 Breast Cancer Diagnosis Prediction

Predict whether a breast tumor is **Benign** or **Malignant** using machine learning.

## 🎯 Project Overview

End-to-end ML project using the **Wisconsin Breast Cancer Diagnostic** dataset:

- Data cleaning & preprocessing
- Feature engineering
- Training & comparing two models
- Model deployment with Flask web app
- Visual analysis & interpretation

## 📊 Dataset

- **Source**: Wisconsin Breast Cancer (Diagnostic)
- **Samples**: 699
- **Features**: 9 cytological characteristics
- **Target**: Benign (0) / Malignant (1)

## 🧹 Preprocessing & Feature Engineering

- Handled missing values in Bare Nuclei
- Removed outliers (Z-score > 3)
- Standard scaling
- **New features**:
  - CellUniformityRatio = Uniformity Size / Uniformity Shape
  - NucleiDensity = Normal Nucleoli / Clump Thickness

## 🤖 Models

| Model              | Type                  | Strengths                        |
|--------------------|-----------------------|----------------------------------|
| MLPClassifier      | Neural Network        | Captures complex patterns        |
| RandomForest       | Ensemble (Trees)      | Feature importance, robustness   |

- 80/20 stratified train-test split
- Models & scaler saved with joblib

## 📈 Results & Visualizations

- Confusion matrices
- Training loss curve (Neural Network)
- Feature importance (Random Forest)
- Correlation heatmap
- Distribution comparison (Benign vs Malignant)

**Key Insight**: Bare Nuclei, Uniformity of Cell Size & Shape are the most discriminative features.

## 🌐 Web Application (Flask)

- Clean & simple input form
- Choose model (Neural Network or Random Forest)
- Instant prediction
- Results + important visualization pages

## 🛠️ Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Flask
- joblib

## Workflow

Data → Cleaning → Feature Engineering → Model Training → Evaluation → Visualization → Flask Deployment

---

A complete demonstration of real-world ML pipeline from raw data to production web app.
