# Train Model pipeline
#
# ## By:
# [Luis Felipe Ospina](https://github.com/1210lfo)
#
# ## Date:
# 2025-03-14
#
# ## Description:
#
# Set all the code for a simple training pipeline using Logistic Regression.
#

# Import libraries
import sys
from pathlib import Path

# from sklearn.svm import SVC # Remove SVC import
import numpy as np  # Import numpy
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Import utility functions - Ensure pipeline_utils.py is accessible in Python's path
try:
    from pipeline_utils import (
        prepare_data,  # Keep the corrected prepare_data
        print_classification_metrics,
        replace_invalid_values,
        summarize_classification,
        train_test_data_split,
        validate_model,
    )
except ImportError:
    print("Error: Could not import pipeline_utils.")
    print("Please ensure pipeline_utils.py is in the same directory or in your Python path.")
    sys.exit()  # Exit the script if import fails


# Configuration parameters
BASELINE_SCORE = 0.7
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
SCORE_METRIC = "recall"
TEST_SIZE = 0.2
# Removed PREDICTION_THRESHOLD as we are using direct predict() for evaluation


# Load data
URL_DATA = "https://github.com/JoseRZapata/Data_analysis_notebooks/raw/refs/heads/main/data/datasets/Clientes_Telcomunicaciones-Churn_data.csv"
churn_df = pd.read_csv(URL_DATA, low_memory=False, na_values="?")

# Data preparation - use utility function
# Keep replace_invalid_values as it handles the ' ' in TotalCharges and "No internet service"
churn_df = replace_invalid_values(churn_df)


# Feature selection based on the notebook
selected_features = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "Contract",
    "SeniorCitizen",
    "PaymentMethod",
    "tenure",
    "MonthlyCharges",
]  # Exclude 'Churn' from features list before splitting

# Prepare data: select features and target, handle NaNs in target
X_features, Y_target = prepare_data(churn_df, selected_features, TARGET_COLUMN)

# Remove duplicates from features as done in the notebook
# Apply this BEFORE splitting the data
combined_data = pd.concat([X_features, Y_target], axis=1)
combined_data.drop_duplicates(inplace=True)

X_features = combined_data[selected_features]
Y_target = combined_data[TARGET_COLUMN]


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_data_split(
    X_features,
    Y_target,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=Y_target,
)

# Define categorical and numerical features based on the selected features
# Need to re-determine based on the potentially smaller feature set
categorical_features = X_features.select_dtypes(include=["object", "category"]).columns
numerical_features = X_features.select_dtypes(include=np.number).columns


# Create preprocessing pipelines for numerical and categorical features
# Aligned with notebook (KNN for numerical, Simple for cat)
numerical_transformer = Pipeline(
    steps=[
        ("imputer", KNNImputer(n_neighbors=5))  # Aligning with notebook
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ("numericas", numerical_transformer, numerical_features),
        (
            "categoricas nominales",
            categorical_transformer,
            categorical_features,
        ),  # Naming convention from notebook
    ]
)


# Create the full pipeline including preprocessing and model training
# UPDATED: Use LogisticRegression with class_weight='balanced'
# model_pipeline = Pipeline(
# steps=[
# ("preprocessor", preprocessor),
# (
# "classifier",
# LogisticRegression(solver="liblinear", C=10, penalty="l1"),
# ),  # Use Logistic Regression
# ]
# )

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                solver="liblinear",
                C=10,
                penalty="l1",
            ),
        ),  # Use Logistic Regression
    ]
)

# Train the model
print("Training Logistic Regression model...")
model_pipeline.fit(x_train, y_train)


# Evaluate the model using standard predict method
print("\nEvaluating model...")

y_pred_train = model_pipeline.predict(x_train)
y_pred_test = model_pipeline.predict(x_test)

# Summarize classification metrics
train_metrics = summarize_classification(y_train, y_pred_train)
test_metrics = summarize_classification(y_test, y_pred_test)


print(f"Train {SCORE_METRIC}: {train_metrics[SCORE_METRIC]:.4f}")
print(f"Test {SCORE_METRIC}: {test_metrics[SCORE_METRIC]:.4f}")

# Print detailed classification metrics
print("\n=== Detailed classification metrics ===")
# The print_classification_metrics function in pipeline_utils already uses zero_division=0
print_classification_metrics(y_test, y_pred_test)


# Model validation using utility function
print("\n=== Model Validation ===")
model_validated = validate_model(
    test_metrics[SCORE_METRIC],
    BASELINE_SCORE,
    metric_name=SCORE_METRIC,
    higher_is_better=True,
)

if not model_validated:
    print("Model does not meet performance requirements!")
    print(f"Required {SCORE_METRIC} >= {BASELINE_SCORE}, but got {test_metrics[SCORE_METRIC]:.4f}")
    raise ValueError("Model-validation-failed")

# Save model
print("\n=== Saving Logistic Regression model ===")
DATA_MODEL = Path.cwd().resolve() / "models"
DATA_MODEL.mkdir(parents=True, exist_ok=True)

# Updated model filename
model_filename = "telco_churn_logistic_regression_model.joblib"
model_path = DATA_MODEL / model_filename

dump(model_pipeline, model_path)

print(f"Model saved successfully at: {model_path}")
