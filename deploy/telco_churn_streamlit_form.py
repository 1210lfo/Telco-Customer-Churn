import os
from pathlib import Path  # Import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

# --- Utility functions adapted from pipeline_utils.py ---
# (Including minimal functions needed for input preprocessing in the app)


def replace_invalid_values_deploy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace invalid values in specific columns with np.nan or other appropriate values,
    specifically for the input data from the Streamlit form.
    Adapted from pipeline_utils.py.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data from the form.

    Returns:
        pd.DataFrame: DataFrame with invalid values replaced.
    """
    # Replace specific string values with np.nan in numeric-like columns
    # The form might not have a text input for TotalCharges directly,
    # but keep this for consistency if the user provides raw data.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)

    # Replace specific string values with np.nan in categorical columns
    # Ensure these columns exist in the form data
    categorical_cols_to_check = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in categorical_cols_to_check:
        if col in df.columns:
            # Handle "No internet service" specifically if it's an option in the form
            df[col] = df[col].replace("No internet service", np.nan)

    return df


# Note: replace_out_of_range_values and prepare_data (full version) are not needed here
# as the form collects specific feature values directly and the pipeline handles
# the rest of the preprocessing like encoding and imputation.

# --- End of adapted utility functions ---


def get_user_data() -> pd.DataFrame:
    """
    Get the data provided by the user through the Streamlit form.
    Creates a DataFrame matching the features used by the model.

    :return: DataFrame with user input data
    """
    user_data = {}

    st.header("Ingrese sus datos sr. cliente:")

    # Using columns for a better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        user_data["gender"] = st.radio("Género:", options=["Male", "Female"], horizontal=True)
        user_data["SeniorCitizen"] = st.radio(
            "Ciudadano Mayor:", options=["Yes", "No"], horizontal=True
        )
        user_data["Partner"] = st.radio("Pareja:", options=["Yes", "No"], horizontal=True)
        user_data["Dependents"] = st.radio("Dependientes:", options=["Yes", "No"], horizontal=True)
        user_data["PhoneService"] = st.radio(
            "Servicio Telefónico:", options=["Yes", "No"], horizontal=True
        )
        user_data["PaperlessBilling"] = st.radio(
            "Facturación Electrónica:", options=["Yes", "No"], horizontal=True
        )

    with col2:
        # Tenure as a number input, align range with data if possible
        user_data["tenure"] = st.slider(
            "Antigüedad (meses):", min_value=0, max_value=72, value=1, step=1
        )
        user_data["MultipleLines"] = st.selectbox(
            "Líneas Múltiples:", options=["No phone service", "No", "Yes"]
        )
        user_data["InternetService"] = st.selectbox(
            "Servicio de Internet:", options=["DSL", "Fiber optic", "No"]
        )
        user_data["OnlineSecurity"] = st.selectbox(
            "Seguridad Online:", options=["No internet service", "No", "Yes"]
        )
        user_data["OnlineBackup"] = st.selectbox(
            "Copia de Seguridad Online:", options=["No internet service", "No", "Yes"]
        )
        user_data["DeviceProtection"] = st.selectbox(
            "Protección del Dispositivo:", options=["No internet service", "No", "Yes"]
        )

    with col3:
        user_data["TechSupport"] = st.selectbox(
            "Soporte Técnico:", options=["No internet service", "No", "Yes"]
        )
        user_data["StreamingTV"] = st.selectbox(
            "Streaming TV:", options=["No internet service", "No", "Yes"]
        )
        user_data["StreamingMovies"] = st.selectbox(
            "Streaming Películas:", options=["No internet service", "No", "Yes"]
        )
        user_data["Contract"] = st.selectbox(
            "Contrato:", options=["Month-to-month", "One year", "Two year"]
        )
        user_data["PaymentMethod"] = st.selectbox(
            "Método de Pago:",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        # Monthly Charges as number input
        user_data["MonthlyCharges"] = st.slider(
            "Cargo Mensual:", min_value=0.0, max_value=200.0, value=50.0, step=0.1
        )
        # Total Charges - This is usually tenure * MonthlyCharges, but let's collect it
        # as a number input for simplicity, or it could be calculated.
        # Based on features used, TotalCharges was in the selected list.
        user_data["TotalCharges"] = st.slider(
            "Cargo Total:", min_value=0.0, value=100.0, step=0.1, max_value=4000.0
        )

    # Convert collected data to DataFrame
    df = pd.DataFrame.from_dict(user_data, orient="index").T

    # --- Preprocessing for consistency with pipeline input ---
    # Apply minimal preprocessing needed for consistency with training data structure
    df = replace_invalid_values_deploy(df)  # Handle potential string NaNs or invalid numbers

    # Convert features collected as 'Yes'/'No' strings to 1/0 or boolean if the pipeline expects it.
    # Check the dtypes of X_features just before the preprocessor in simple_train_pipeline.py
    # SeniorCitizen was numeric (int64) in the notebook features. Map Yes/No to 1/0 and keep as int.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({"Yes": 1, "No": 0}).astype(int)
    # Partner, Dependents, PhoneService, PaperlessBilling were potentially treated as boolean or
    # objects.
    # Let's map them to Yes/No strings as the OneHotEncoder handles these categories.
    # If the form uses Yes/No strings, they can be passed directly to the pipeline's OneHotEncoder
    # for Partner, Dependents, PhoneService, PaperlessBilling.

    # Ensure numerical features are float
    numerical_cols_input = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numerical_cols_input:
        if col in df.columns:
            # Coerce errors to NaN, which the pipeline's imputer will handle
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # The order of columns in the input DataFrame should match the order of features the
    # ColumnTransformer in the pipeline expects. Get the list of features used in the pipeline.
    # Re-listing the features used in the Logistic Regression simple_train_pipeline.py
    final_selected_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]
    # Reindex the DataFrame to match the expected column order and ensure all features are present
    # Fill potentially missing columns (if any were not in the form) with NaN - the pipeline's
    # imputer will handle them.
    # This also ensures the order is correct for the pipeline.
    df = df.reindex(columns=final_selected_features)

    return df


# --- Model Loading ---
# Define the expected location of the model file based on the project structure
# Project root: Telco-Customer-Churn
# Script location: Telco-Customer-Churn/src/deploy/
# Model location: Telco-Customer-Churn/models/
# To get from script to model: go up two levels (..) (..)
# from script to project root, then down to models/
project_root = Path(__file__).parent.parent.parent
# CORRECTED: Remove the extra "Telco-Customer-Churn" from the path
model_dir = project_root / "models"
model_name = "telco_churn_logistic_regression_model.joblib"  # Correct model filename
model_path = str(model_dir / model_name)


# --- Load the model ---
# Ensure the load_model function is defined elsewhere in the script (as in previous versions)
@st.cache_resource
def load_model(model_file_path: str) -> Pipeline:
    """
    Loads a model in joblib format (.joblib extension).

    Args:
        model_file_path (str): The absolute path where the trained model is stored.

    Returns:
        Pipeline: The trained model, a scikit-learn Pipeline object.
    """
    if not os.path.exists(model_file_path):
        st.error(f"Error: Archivo del modelo no encontrado en: {model_file_path}")
        st.stop()  # Stop the app if model is not found

    with st.spinner("Cargando modelo..."):
        model = load(model_file_path)

    return model


def main() -> None:
    # --- Streamlit App Configuration ---
    st.set_page_config(page_title="Predicción de Abandono de Clientes Telco", layout="wide")

    # --- Model Loading ---
    model_dir = Path(__file__).parent.parent / "models"
    model_name = "telco_churn_logistic_regression_model.joblib"  # Correct model filename
    model_path = str(model_dir / model_name)

    # Call load_model with the corrected path
    model_pipeline = load_model(model_file_path=model_path)

    # --- App Title and Header ---
    # st.image("path/to/your/telco/image.jpg", use_column_width=True)
    # Optional: Add a relevant image
    st.title("Predicción de Abandono de Clientes de Telecomunicaciones")
    st.markdown("#### Modelo de Regresión Logística")
    st.write("Ingrese los datos del cliente para predecir si es probable que abandone el servicio.")

    # --- Get User Input ---
    df_user_data = get_user_data()

    # --- Make Prediction ---
    predict_button = st.button("Realizar Predicción")

    if predict_button:
        if df_user_data.empty:
            st.warning("Por favor, ingrese los datos del cliente.")
        else:
            # The pipeline expects the same features in the same format as training
            # The get_user_data function aims to create this structure.
            try:
                # Make prediction - the pipeline handles all preprocessing
                prediction = model_pipeline.predict(df_user_data)
                prediction_proba = model_pipeline.predict_proba(df_user_data)[
                    :, 1
                ]  # Probability of churn (class 1)

                # --- Display Results ---
                st.write("---")
                st.subheader("Resultado de la Predicción:")

                churn_status = "Sí" if prediction[0] == 1 else "No"
                probability = prediction_proba[0] * 100

                st.write(
                    f"Basado en los datos ingresados, la predicción es que el cliente "
                    f"**{churn_status}** abandonará el servicio."
                )
                st.write(f"Probabilidad estimada de abandono: **{probability:.2f}%**")

                # Display a more user-friendly message based on prediction
                if prediction[0] == 1:
                    st.error("Predicción: ¡Alto riesgo de abandono!")
                else:
                    st.success("Predicción: Bajo riesgo de abandono.")

                st.write("---")
                st.write("Nota: Esta predicción se basa en el modelo entrenado y sus datos.")

            except Exception as e:
                st.error(f"Ocurrió un error al realizar la predicción: {e}")
                st.write("Por favor, verifique los datos ingresados.")


if __name__ == "__main__":
    main()
