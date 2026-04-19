# ============================================================
# A1, A2, A3 - STACKING + PIPELINE + LIME
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# FUNCTION 1: Load dataset
# ============================================================
def load_dataset(file_path):
    return pd.read_excel(file_path)


# ============================================================
# FUNCTION 2: Clean dataset
# ============================================================
def clean_dataset(data_frame):
    data_frame = data_frame.copy()
    data_frame = data_frame.drop_duplicates()
    data_frame.columns = data_frame.columns.str.strip()
    return data_frame


# ============================================================
# FUNCTION 3: Split features and target
# ============================================================
def split_features_target(data_frame, target_column):
    x_data = data_frame.drop(columns=[target_column])
    y_data = data_frame[target_column]
    return x_data, y_data


# ============================================================
# FUNCTION 4: Column types
# ============================================================
def get_column_types(x_data):
    num_cols = x_data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = x_data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


# ============================================================
# FUNCTION 5: Preprocessor
# ============================================================
def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])


# ============================================================
# FUNCTION 6: Stacking model
# ============================================================
def build_stacking_model(problem_type):

    if problem_type == "classification":
        return StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier()),
                ("gb", GradientBoostingClassifier()),
                ("et", ExtraTreesClassifier())
            ],
            final_estimator=LogisticRegression(max_iter=2000)
        )

    else:
        return StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor()),
                ("gb", GradientBoostingRegressor()),
                ("et", ExtraTreesRegressor())
            ],
            final_estimator=LinearRegression()
        )


# ============================================================
# FUNCTION 7: Pipeline
# ============================================================
def build_pipeline(preprocessor, model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# ============================================================
# MAIN PROGRAM (AFTER ALL FUNCTIONS ✅)
# ============================================================
if __name__ == "__main__":

    # ✅ FIXED FILE PATH
    file_path = r"C:\Users\preet\Downloads\classData.xlsx"

    # Load data
    df = load_dataset(file_path)
    df = clean_dataset(df)

    print("\nDataset Loaded")
    print(df.head())

    # User input
    target_column = input("\nEnter target column: ").strip()
    problem_type = input("Enter problem type (classification/regression): ").strip()

    # Split
    X, y = split_features_target(df, target_column)

    # Encode target if needed
    if problem_type == "classification" and y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Preprocessing
    num_cols, cat_cols = get_column_types(X)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Model
    model = build_stacking_model(problem_type)

    # Pipeline
    pipe = build_pipeline(preprocessor, model)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    pipe.fit(X_train, y_train)

    print("\nModel trained successfully")

    # Evaluate
    y_pred = pipe.predict(X_test)

    if problem_type == "classification":
        print("\nAccuracy:", accuracy_score(y_test, y_pred))
        print("\nReport:\n", classification_report(y_test, y_pred))
    else:
        print("\nMAE:", mean_absolute_error(y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
        print("R2:", r2_score(y_test, y_pred))

    # ======================================================
    # LIME
    # ======================================================
    X_train_tf = pipe.named_steps["preprocessor"].transform(X_train)
    X_test_tf = pipe.named_steps["preprocessor"].transform(X_test)

    if hasattr(X_train_tf, "toarray"):
        X_train_tf = X_train_tf.toarray()
        X_test_tf = X_test_tf.toarray()

    explainer = LimeTabularExplainer(
        training_data=X_train_tf,
        feature_names=X.columns.tolist(),
        mode=problem_type
    )

    if problem_type == "classification":
        predict_fn = pipe.named_steps["model"].predict_proba
    else:
        predict_fn = pipe.named_steps["model"].predict

    exp = explainer.explain_instance(X_test_tf[0], predict_fn)

    print("\nLIME Explanation:")
    for i in exp.as_list():
        print(i)

    exp.show_in_notebook(show_table=True)
