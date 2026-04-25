# ============================================================
# A1 to A5 Full Correct Code (MODIFIED PATH)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer

# SHAP optional
try:
    import shap
    shap_available = True
except:
    shap_available = False
    print("SHAP is not installed. SHAP part will be skipped.")

# ============================================================
# LOAD DATASET (FIXED HERE)
# ============================================================

DATASET_PATH = "classData.xlsx"   # ✅ PLACE FILE IN SAME FOLDER

import os
if not os.path.exists(DATASET_PATH):
    print("❌ File not found. Check file location.")
    exit()

df = pd.read_excel(DATASET_PATH)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# ============================================================
# TARGET AND FEATURES
# ============================================================

target_columns = ['G', 'C', 'B', 'A']
feature_columns = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

X = df[feature_columns]

df["Fault_Type"] = df[target_columns].astype(str).agg("".join, axis=1)
y = df["Fault_Type"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# MODEL FUNCTION
# ============================================================

def evaluate_model(model, Xtr, Xte, ytr, yte, title):
    print("\n===================================================")
    print(title)
    print("===================================================")

    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)

    acc = accuracy_score(yte, y_pred)

    print("Accuracy:", round(acc * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(yte, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(yte, y_pred))

    return acc, model

# ============================================================
# A1 CORRELATION
# ============================================================

corr_matrix = X.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

baseline_model = XGBClassifier(eval_metric="mlogloss")
baseline_acc, baseline_model = evaluate_model(
    baseline_model, X_train_scaled, X_test_scaled, y_train, y_test,
    "Baseline Model"
)

# ============================================================
# A2 PCA 99%
# ============================================================

pca_99 = PCA(n_components=0.99)

X_train_pca_99 = pca_99.fit_transform(X_train_scaled)
X_test_pca_99 = pca_99.transform(X_test_scaled)

pca99_model = XGBClassifier(eval_metric="mlogloss")
pca99_acc, _ = evaluate_model(
    pca99_model, X_train_pca_99, X_test_pca_99, y_train, y_test,
    "PCA 99%"
)

# ============================================================
# A3 PCA 95%
# ============================================================

pca_95 = PCA(n_components=0.95)

X_train_pca_95 = pca_95.fit_transform(X_train_scaled)
X_test_pca_95 = pca_95.transform(X_test_scaled)

pca95_model = XGBClassifier(eval_metric="mlogloss")
pca95_acc, _ = evaluate_model(
    pca95_model, X_train_pca_95, X_test_pca_95, y_train, y_test,
    "PCA 95%"
)

# ============================================================
# A4 FEATURE SELECTION
# ============================================================

sfs = SequentialFeatureSelector(
    RandomForestClassifier(),
    n_features_to_select=3,
    direction="forward",
    cv=3
)

sfs.fit(X_train_scaled, y_train)

X_train_sfs = sfs.transform(X_train_scaled)
X_test_sfs = sfs.transform(X_test_scaled)

sfs_model = XGBClassifier(eval_metric="mlogloss")
sfs_acc, _ = evaluate_model(
    sfs_model, X_train_sfs, X_test_sfs, y_train, y_test,
    "Feature Selection"
)

# ============================================================
# A5 LIME
# ============================================================

explainer = LimeTabularExplainer(
    X_train_scaled,
    feature_names=feature_columns,
    class_names=[str(c) for c in label_encoder.classes_],
    mode="classification"
)

exp = explainer.explain_instance(
    X_test_scaled[0],
    baseline_model.predict_proba
)

exp.save_to_file("lime_explanation.html")

print("LIME saved")

# ============================================================
# SHAP
# ============================================================

if shap_available:
    explainer = shap.TreeExplainer(baseline_model)
    shap_values = explainer.shap_values(X_test_scaled)
    shap.summary_plot(shap_values, X_test_scaled)
else:
    print("Install SHAP: pip install shap")

# ============================================================
# FINAL RESULTS
# ============================================================

results = pd.DataFrame({
    "Method": ["Baseline", "PCA 99", "PCA 95", "SFS"],
    "Accuracy": [
        baseline_acc * 100,
        pca99_acc * 100,
        pca95_acc * 100,
        sfs_acc * 100
    ]
})

print(results)
