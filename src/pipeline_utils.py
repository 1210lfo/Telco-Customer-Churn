"""
Telco Customer Churn Pipeline Utilities

Este módulo proporciona funciones de utilidad para el preprocesamiento de datos,
ingeniería de características y evaluación de modelos para el pipeline de
predicción de abandono de clientes de telecomunicaciones.

Autor: Luis Felipe Ospina
Fecha: 23 de abril de 2025
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Corrected: Import train_test_split to be used within train_test_data_split
from sklearn.model_selection import train_test_split

# Definición de constantes
MAX_MONTHLY_CHARGE = 500
MIN_MONTHLY_CHARGE = 0


def replace_out_of_range_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza los valores fuera de rango en columnas numéricas con np.nan.

    Args:
        X (pd.DataFrame): DataFrame de entrada que contiene los datos a procesar.

    Returns:
        pd.DataFrame: DataFrame con los valores fuera de rango reemplazados por np.nan.

    Notas:
        Esta función reemplaza los valores en 'MonthlyCharges' que están fuera del
        rango definido por MIN_MONTHLY_CHARGE y MAX_MONTHLY_CHARGE con np.nan.
    """
    # Asegurarse de que la entrada sea un pandas DataFrame
    assert isinstance(X, pd.DataFrame)

    # Crear una máscara para los valores fuera del rango definido para 'MonthlyCharges'
    mask = ~(
        (X["MonthlyCharges"] > MIN_MONTHLY_CHARGE)
        & (X["MonthlyCharges"] < MAX_MONTHLY_CHARGE)
    )

    # Reemplazar los valores fuera de rango con np.nan
    X.loc[mask, "MonthlyCharges"] = np.nan

    return X


def replace_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza valores inválidos en columnas específicas con np.nan u otros valores apropiados.

    Args:
        df (pd.DataFrame): DataFrame de entrada que contiene los datos a procesar.

    Returns:
        pd.DataFrame: DataFrame con los valores inválidos reemplazados.
    """
    # Reemplazar valores de string específicos con np.nan en columnas de tipo numérico
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)

    # Reemplazar valores de string específicos con np.nan en columnas categóricas
    for col in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        df[col] = df[col].replace("No internet service", np.nan)

    return df


def prepare_data(
    df: pd.DataFrame, features: List[str], target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara los datos para el modelado seleccionando características y la columna objetivo,
    eliminando filas con valores faltantes en la columna objetivo y convirtiéndola a un
    formato numérico.

    Args:
        df (pd.DataFrame): El DataFrame de entrada que contiene los datos.
        features (List[str]): Una lista de nombres de columnas de características a utilizar.
        target_column (str): El nombre de la columna de la variable objetivo.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Una tupla que contiene el DataFrame de características (X)
                                         y la Serie objetivo (Y).
    """
    # Seleccionar características y objetivo
    # Incluimos la columna objetivo aquí temporalmente para manejar sus NaNs
    cols_to_select = features + [target_column]
    filtered_df = df[cols_to_select].copy()

    # Convertir la columna objetivo a numérica (0 para 'No', 1 para 'Yes')
    # Asegurarse de que los valores sean strings antes de mapear para evitar errores con NaNs u otros tipos
    filtered_df[target_column] = (
        filtered_df[target_column].astype(str).map({"No": 0, "Yes": 1})
    )

    # Eliminar filas donde la columna objetivo tiene valores faltantes después del mapeo
    # Esto manejará cualquier valor original que no fuera 'No' o 'Yes' y se convirtió en NaN
    filtered_df.dropna(subset=[target_column], inplace=True)

    # Convertir la columna objetivo a tipo entero después de eliminar NaNs
    filtered_df[target_column] = filtered_df[target_column].astype(int)

    # Separar características y objetivo
    X_features = filtered_df[features]
    Y_target = filtered_df[target_column]

    return X_features, Y_target


# Corrected: Added stratify to the function definition
def train_test_data_split(
    X: pd.DataFrame,
    Y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: pd.Series = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos de características y objetivo en conjuntos de entrenamiento y prueba.

    Args:
        X (pd.DataFrame): El DataFrame de características.
        Y (pd.Series): La Serie objetivo.
        test_size (float): La proporción del conjunto de datos a incluir en la división de prueba.
                           Por defecto es 0.2.
        random_state (int): Controla el barajado aplicado a los datos antes de aplicar
                            la división. Por defecto es 42.
        stratify (pd.Series, optional): Si no es None, los datos se dividen de forma estratificada,
                                        usando Y como las etiquetas de clase. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Una tupla que contiene
                                                                  x_train, x_test,
                                                                  y_train, y_test.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return x_train, x_test, y_train, y_test


def summarize_classification(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas clave de clasificación.

    Args:
        y_true (pd.Series): Los valores objetivo verdaderos.
        y_pred (np.ndarray): Los valores objetivo predichos.

    Returns:
        Dict[str, float]: Un diccionario que contiene las métricas calculadas: accuracy,
                          precision, recall y f1_score.
    """
    # Added zero_division=0 to handle cases where a class has no predicted samples
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
    return metrics


def print_classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """
    Imprime un informe detallado de clasificación.

    Args:
        y_true (pd.Series): Los valores objetivo verdaderos.
        y_pred (np.ndarray): Los valores objetivo predichos.
    """
    # Added zero_division=0 to handle cases where a class has no predicted samples
    print(classification_report(y_true, y_pred, zero_division=0))


def validate_model(
    metric_result: float,
    baseline_score: float,
    metric_name: str = "recall",
    higher_is_better: bool = True,
) -> bool:
    """
    Valida si la métrica de rendimiento del modelo cumple con una línea base especificada.

    Args:
        metric_result (float): El valor real de la métrica alcanzado por el modelo.
        baseline_score (float): La puntuación mínima (o máxima) aceptable para la métrica.
        metric_name (str): El nombre de la métrica que se está validando (para impresión). Por defecto es "recall".
        higher_is_better (bool): Indica si un valor más alto para la métrica es mejor (ej. accuracy, recall)
                                 o si un valor más bajo es mejor (ej. tasa de error). Por defecto es True.

    Returns:
        bool: True si el resultado de la métrica del modelo cumple o excede la línea base (o está por debajo si higher_is_better es False),
              False en caso contrario.

    Notas:
        Esta función compara el rendimiento del modelo en una métrica específica con una línea base predefinida.
        También imprime un mensaje indicando si la validación pasó o falló.
    """
    # Determinar si la validación pasa en función de si una puntuación más alta es mejor
    if higher_is_better:
        validation_passed = metric_result > baseline_score
    else:
        validation_passed = metric_result < baseline_score

    # Imprimir el resultado de la validación
    if validation_passed:
        print(
            f"Validación del modelo pasada: {metric_name} = {metric_result:.4f} "
            f"(línea base: {baseline_score:.4f})"
        )
    else:
        print(
            f"Validación del modelo fallida: {metric_name} = {metric_result:.4f} "
            f"(línea base: {baseline_score:.4f})"
        )

    return validation_passed
