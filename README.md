# Telco Customer Churn - Proyecto de Ciencia de Datos End-to-End

## 📋 Descripción del Proyecto

Este es un proyecto completo de ciencia de datos que desarrolla un modelo de machine learning para predecir el abandono de clientes (churn) en una empresa de telecomunicaciones. El proyecto implementa toda la pipeline desde la exploración de datos hasta el despliegue de modelos en producción.

### 🎯 Objetivos

- **Desarrollo End-to-End**: Implementar una solución completa desde la exploración de datos hasta el despliegue
- **Predicción de Churn**: Identificar clientes con alta probabilidad de abandonar el servicio
- **Buenas Prácticas**: Aplicar metodologías de desarrollo de software y MLOps
- **Colaboración**: Fomentar el trabajo en equipo mediante Pull Requests y metodologías ágiles

## 🚀 Aplicaciones Desplegadas

### 📝 Predicción Individual (Formulario)

**URL**: <https://telco-customer-churn-xzdctfimazaa4rd2jcfvx3.streamlit.app/>

Interfaz web para realizar predicciones individuales ingresando los datos de un cliente específico a través de un formulario interactivo.

### 📊 Predicción en Lote (Batch)

**URL**: <https://telco-customer-churn-vv39jckjelbc5yyq9gd56y.streamlit.app/>

Aplicación para procesar múltiples clientes simultáneamente mediante la carga de archivos CSV y descarga de resultados.

## 📁 Estructura del Proyecto

```
Telco-Customer-Churn/
├── .github/                    # Configuración CI/CD y workflows
├── data/                       # Datos del proyecto
│   ├── external/              # Datos de fuentes externas
│   ├── interim/               # Datos intermedios procesados
│   ├── processed/             # Datos listos para ML
│   └── raw/                   # Datos originales sin procesar
├── deploy/                     # Aplicaciones de despliegue
│   ├── telco_churn_streamlit_form.py    # App individual
│   └── telco_churn_streamlit_batch.py   # App por lotes
├── docs/                       # Documentación y reportes
├── models/                     # Modelos entrenados
│   └── telco_churn_logistic_regression_model.joblib
├── notebooks/                  # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/                        # Código fuente
│   └── telco_churn/           # Módulo principal
├── tests/                      # Tests unitarios
├── .gitignore                  # Archivos ignorados por Git
├── .pre-commit-config.yaml     # Configuración pre-commit hooks
├── .python-version             # Versión de Python
├── pyproject.toml              # Configuración del proyecto
└── README.md                   # Este archivo
```

## 🔍 Descripción de los Datos

El dataset utiliza el conjunto **Telco Customer Churn** que contiene información de clientes con las siguientes características:

### Variables Objetivo

- **Churn**: Indica si el cliente abandonó el servicio (Sí/No)

### Características del Cliente

- **Demografía**: Género, edad (SeniorCitizen), estado civil (Partner), dependientes
- **Servicios**: Telefonía, internet, múltiples líneas, seguridad online, backup, etc.
- **Cuenta**: Tipo de contrato, método de pago, facturación, antigüedad (tenure)
- **Financiero**: Cargos mensuales (MonthlyCharges) y totales (TotalCharges)

**Fuentes**:

- [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM Community Dataset](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

## ⚙️ Configuración del Entorno

### Requisitos

- **Python**: 3.11+
- **Gestor de paquetes**: UV (recomendado)

### Instalación

1. **Instalar UV** (si no lo tienes):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clonar el repositorio**:

```bash
git clone https://github.com/tu-usuario/Telco-Customer-Churn.git
cd Telco-Customer-Churn
```

3. **Crear entorno y sincronizar dependencias**:

```bash
uv venv --python 3.11
uv sync
```

4. **Activar el entorno virtual**:

```bash
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

### Ejecutar aplicaciones localmente

**Formulario individual**:

```bash
streamlit run deploy/telco_churn_streamlit_form.py
```

**Predicción en lote**:

```bash
streamlit run deploy/telco_churn_streamlit_batch.py
```

## 🛠️ Desarrollo

### Herramientas de Calidad de Código

- **Linting**: Ruff
- **Formateo**: Ruff Format  
- **Tipado estático**: MyPy
- **Pre-commit hooks**: Validación automática antes de commits

### Comandos útiles

```bash
# Instalar hooks de pre-commit
uv run pre-commit install

# Ejecutar linting y formateo
uv run ruff check .
uv run ruff format .

# Verificar tipos
uv run mypy src/

# Ejecutar tests
uv run pytest tests/
```

## 🤝 Contribución

1. Fork del repositorio
2. Crear rama para feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

La rama principal está protegida y requiere:

- ✅ Revisión de al menos 2 personas
- ✅ Pasar todos los checks de CI/CD
- ✅ Resolución de conflictos

## 📊 Modelo

- **Algoritmo**: Regresión Logística
- **Pipeline**: Incluye preprocesamiento y transformación de datos
- **Métricas**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Formato**: Joblib para serialización

## 📄 Licencia

Este proyecto está desarrollado con fines educativos y de demostración de buenas prácticas en ciencia de datos.

---

**Desarrollado con ❤️ para demostrar un pipeline completo de Machine Learning**
