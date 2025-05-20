import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from matplotlib.backends.backend_pdf import PdfPages
import os
import platform
import subprocess
from flask import Flask, render_template, request
 
# -------- Helper function to open PDF automatically --------
def open_file(filepath):
    if platform.system() == 'Windows':
        os.startfile(filepath)
    elif platform.system() == 'Darwin':
        os.system(f"open '{filepath}'")
    else:
        os.system(f"xdg-open '{filepath}'")
 
# -------- 1. Load and preprocess data --------
df = pd.read_csv("water_potability.csv")
df = df.fillna(df.median())
 
# -------- 2. Prepare data for modeling --------
X = df.drop("Potability", axis=1)
y = df["Potability"]
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
 
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)
 
# -------- 3. Train Random Forest with GridSearch --------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
 
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
 
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
 
accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = round(accuracy * 100, 2)
 
# -------- 4. Create PDF report --------
pdf_path = "Water_Quality_Classification_Report.pdf"
with PdfPages(pdf_path) as pdf:
 
    # --- Title Page ---
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.5, 0.6, "Water Quality Classification Report", fontsize=28, fontweight='bold', ha='center')
    plt.text(0.5, 0.4, f"Random Forest Accuracy: {accuracy_percent}%", fontsize=24, ha='center', color='navy')
    pdf.savefig()
    plt.close()
 
    # --- Potability Distribution with Percentages ---
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x="Potability", data=df)
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.2f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, color='black')
    plt.title("Potability Distribution with Percentages")
    pdf.savefig()
    plt.close()
 
    # --- All Feature Distributions ---
    feature_cols = df.columns[:-1]
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()
 
    for i, col in enumerate(feature_cols):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
 
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
 
    fig.suptitle("Feature Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig()
    plt.close()
 
    # --- Correlation Heatmap ---
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
 
    # --- Confusion Matrix ---
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    pdf.savefig()
    plt.close()
 
    # --- Feature Importances ---
    importances = best_model.feature_importances_
    features = X.columns
 
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title("Feature Importances from Random Forest")
    pdf.savefig()
    plt.close()
 
    # --- Classification Report Page ---
    report_str = classification_report(y_test, y_pred)
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.01, 0.99, "Classification Report", fontsize=20, fontweight='bold', va='top')
    plt.text(0.01, 0.95, report_str, fontsize=12, family='monospace', va='top')
    pdf.savefig()
    plt.close()
 
print(f"PDF report saved as {pdf_path}")
 
# -------- 5. Open PDF automatically --------
open_file(pdf_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
   app.run(debug=False)

 
