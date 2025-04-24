import os
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

# --- Utility functions adapted from telco_churn_streamlit.py ---
# (Including minimal functions needed for input preprocessing in the app)


# Reusing the preprocessing logic from the forms app, adapted to handle a DataFrame
def preprocess_telco_data_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess batch Telco data from a CSV file to match the format expected by the model pipeline.
    Adapts the preprocessing logic from get_user_data in the forms app.

    Args:
        df (pd.DataFrame): The original dataframe loaded from CSV.

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for prediction.
    """
    processed_df = df.copy()

    # Apply invalid values replacement logic from the forms app
    # This handles potential ' ' in TotalCharges and ensures numerical conversion robustness
    if "TotalCharges" in processed_df.columns:
        processed_df["TotalCharges"] = pd.to_numeric(processed_df["TotalCharges"], errors="coerce")
        # No need to replace " " specifically after using to_numeric with coerce

    # Convert 'Yes'/'No' inputs for SeniorCitizen to 1.0/0.0 (float)
    # Handle potential missing values in this column gracefully
    if "SeniorCitizen" in processed_df.columns:
        processed_df["SeniorCitizen"] = processed_df["SeniorCitizen"].map(
            {
                "Yes": 1.0,
                "No": 0.0,
            }
        )
        # If there are NaNs after mapping (e.g., blank cells), leave them as NaN for the imputer
        processed_df["SeniorCitizen"] = processed_df["SeniorCitizen"].astype(float)

    # Ensure numerical features are float.
    # Handle potential missing values (empty cells) using errors='coerce'
    numerical_cols_input = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numerical_cols_input:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce").astype(float)

    # Ensure categorical features are object/string type before the pipeline's OneHotEncoder
    # This is crucial if some columns might be read as bool or other types by pd.read_csv
    categorical_cols_to_ensure_string = [
        "gender",
        "Partner",
        "Dependents",
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
    ]
    for col in categorical_cols_to_ensure_string:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(str)  # Convert to string to be safe

    # Reindex the DataFrame to match the expected column order and ensure all features are present
    # Use the exact list of features the model pipeline was trained on.
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
    # Ensure all features are present and in the correct order.
    # Fill missing columns (if the uploaded CSV is missing a feature column) with NaN.
    # The pipeline's imputer will handle these NaNs.
    processed_df = processed_df.reindex(columns=final_selected_features)

    return processed_df


@st.cache_resource  # Cache the model loading
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
    st.set_page_config(page_title="Predicción de Abandono en Lote", layout="wide")

    # --- Model Loading ---
    model_dir = Path(__file__).parent.parent / "models"
    model_name = "telco_churn_logistic_regression_model.joblib"  # Correct model filename
    model_path = str(model_dir / model_name)

    model_pipeline = load_model(model_file_path=model_path)

    # --- App Title and Header ---
    st.title("Predicción de Abandono de Clientes (Procesamiento por Lote)")
    st.markdown("#### Modelo de Regresión Logística")
    st.write(
        "Suba un archivo CSV con los datos de los clientes para obtener predicciones de abandono."
    )

    # --- Batch Prediction Interface ---
    st.subheader("Subir archivo CSV para predicción en lote")

    # File uploader allows user to upload a CSV file
    uploaded_file = st.file_uploader("Seleccione un archivo CSV", type="csv")

    if uploaded_file is not None:
        # Load the uploaded CSV into a pandas DataFrame
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Previsualización de los datos cargados:")
            st.dataframe(df_batch.head())

            # Add a button to trigger prediction
            if st.button("Realizar Predicciones en Lote"):
                with st.spinner("Procesando datos y realizando predicciones..."):
                    try:
                        # Preprocess the batch data
                        df_processed_batch = preprocess_telco_data_batch(df_batch.copy())
                        # Use .copy() to avoid modifying original df_batch in case it's needed later

                        # Make predictions using the loaded pipeline
                        # The pipeline handles all further preprocessing
                        # (imputation, encoding, scaling)
                        predictions = model_pipeline.predict(df_processed_batch)
                        # Optionally get probabilities
                        predictions_proba = model_pipeline.predict_proba(df_processed_batch)[
                            :, 1
                        ]  # Probability of churn (class 1)

                        # Add predictions to the original DataFrame or processed DataFrame
                        result_df = (
                            df_batch.copy()
                        )  # Add predictions to a copy of the original uploaded data
                        result_df["Predicted_Churn"] = predictions
                        result_df["Predicted_Churn_Probability"] = predictions_proba
                        result_df["Churn_Status"] = result_df["Predicted_Churn"].map(
                            {
                                0: "No Churn",
                                1: "Churn",
                            }
                        )

                        # Display results
                        st.success("Predicciones completadas!")
                        st.subheader("Resultados de la Predicción:")
                        st.dataframe(result_df)

                        # Option to download results as CSV
                        csv_output = result_df.to_csv(index=False)
                        st.download_button(
                            label="Descargar Resultados CSV",
                            data=csv_output,
                            file_name="telco_churn_predictions_batch.csv",
                            mime="text/csv",
                        )

                    except Exception as e:
                        st.error(f"Ocurrió un error durante el procesamiento o la predicción: {e}")
                        st.write("Por favor, verifique el formato y los datos en su archivo CSV.")
                        # Optional: Print details about the error for debugging
                        # import traceback
                        # st.text(traceback.format_exc())

        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            st.info(
                "Por favor, asegúrese de que el archivo es un CSV válido y es codificado en UTF-8."
            )

    else:
        st.info("Por favor, suba un archivo CSV para iniciar la predicción en lote.")

        # Optional: Show expected CSV format or sample data
        st.subheader("Formato de archivo CSV esperado:")
        st.write("Su archivo CSV debe contener las siguientes columnas:")
        st.write(
            ", ".join(
                [
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
            )
        )
        st.write("Asegúrese de que los nombres de las columnas coincidan exactamente.")
        st.write("(incluyendo mayúsculas/minúsculas)")
        # Show a sample DataFrame structure
        # sample_data_batch = pd.DataFrame({ ... sample row data ... })
        # st.dataframe(sample_data_batch)


if __name__ == "__main__":
    main()
