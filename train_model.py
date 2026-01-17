import pandas as pd  # for data handling
import numpy as np   # for numerical operations
import matplotlib.pyplot as plt  # for plotting graphs
import seaborn as sns  # for advanced plotting
import joblib  # for saving/loading models
import os  # for handling file system operations

# Sklearn imports for ML tasks
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import zscore  # for outlier detection

def train_and_save_models():
    df = pd.read_csv("DATA.csv") 

    # Handle Missing Values (Data Cleaning) 
    df["BareNuc"] = pd.to_numeric(df["BareNuc"], errors='coerce')
    df.dropna(inplace=True)  # Remove any rows with missing values

    # Format Data
    df["BareNuc"] = df["BareNuc"].astype(int)
    df["Class"] = df["Class"].replace({2: 0, 4: 1})  # Convert class labels

    # Feature Engineering: Create new meaningful features
    df["CellUniformityRatio"] = df["UnifSize"] / df["UnifShape"]
    df["NucleiDensity"] = df["NormNucl"] / df["Clump"]

    # Outlier Detection and Removal using Z-score
    z_scores = np.abs(df.select_dtypes(include=[np.number]).apply(zscore))
    df = df[(z_scores < 3).all(axis=1)]  # Keep only normal rows

    # Feature Selection
    features = ["Clump", "UnifSize", "UnifShape", "MargAdh", "SingEpiSize",
                "BareNuc", "BlandChrom", "NormNucl", "Mit",
                "CellUniformityRatio", "NucleiDensity"]
    X = df[features]
    y = df["Class"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Train Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)
    nn_model.fit(X_train, y_train)
    y_pred_nn = nn_model.predict(X_test)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Save models and scaler
    os.makedirs("model", exist_ok=True)
    joblib.dump(nn_model, "model/cancer_model.pkl")
    joblib.dump(rf_model, "model/random_forest_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    # Create directory for visualizations
    os.makedirs("static", exist_ok=True)

    # Confusion Matrix for Neural Network
    cm_nn = confusion_matrix(y_test, y_pred_nn)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.title('Neural Network - Confusion Matrix')
    plt.savefig('static/confusion_matrix.png')
    plt.close()

    # Training Loss Curve (NN)
    plt.figure()
    plt.plot(nn_model.loss_curve_)
    plt.title('Neural Network - Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('static/accuracy_graph.png')
    plt.close()

    # Confusion Matrix for Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.title('Random Forest - Confusion Matrix')
    plt.savefig('static/rf_confusion_matrix.png')
    plt.close()

    # Random Forest Feature Importance Plot
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(features)), importances[sorted_idx], align='center')
    plt.yticks(range(len(features)), [features[i] for i in sorted_idx])
    plt.title('Random Forest - Feature Importances')
    plt.tight_layout()
    plt.savefig('static/rf_feature_importance.png')
    plt.close()

    # Histograms for Feature Distributions
    df[features].hist(bins=15, figsize=(12, 10), color='skyblue', edgecolor='black')
    plt.suptitle('Histogram of Features')
    plt.tight_layout()
    plt.savefig('static/feature_histograms.png')
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df[features + ['Class']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('static/correlation_heatmap.png')
    plt.close()

    # Boxplot by Class
    plt.figure(figsize=(12, 8))
    melted = df.melt(id_vars="Class", value_vars=features)
    sns.boxplot(x="variable", y="value", hue="Class", data=melted)
    plt.xticks(rotation=45)
    plt.title("Feature Distributions by Class")
    plt.tight_layout()
    plt.savefig('static/boxplot_by_class.png')
    plt.close()

    # Save Evaluation Metrics to Text File
    metrics_text = f"""
Neural Network Evaluation:
--------------------------
Accuracy:  {accuracy_score(y_test, y_pred_nn):.4f}
Precision: {precision_score(y_test, y_pred_nn):.4f}
Recall:    {recall_score(y_test, y_pred_nn):.4f}
F1 Score:  {f1_score(y_test, y_pred_nn):.4f}

Random Forest Evaluation:
--------------------------
Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}
Precision: {precision_score(y_test, y_pred_rf):.4f}
Recall:    {recall_score(y_test, y_pred_rf):.4f}
F1 Score:  {f1_score(y_test, y_pred_rf):.4f}
"""
    with open("static/metrics.txt", "w") as f:
        f.write(metrics_text)

    print("✅ Models, Metrics, and Visualizations saved successfully!")

# Trigger training when script runs
if __name__ == "__main__":
    train_and_save_models()
