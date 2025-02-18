# Telco customer Chrun Data Science Project 

## Objetivos del proyecto
- **Desarrollo Básico de Inicio a Final**: Implementar una prueba de concepto (POC).
- **Trabajo en Equipo**: Fomentar la colaboración mediante la creación y revisión de Pull Requests (PR) y la aplicación de metodologías ágiles, este proyecto protege la rama principal de forma que solo pueda agregarse información mediante un merge que haya resultado de un pull request y que esté aceptado por mínimo 2 personas.
- **Buenas Prácticas de Desarrollo de Software**:
  - Gestión de ambientes virtuales.
  - Gestión de dependencias.
  - Uso de herramientas de linting, formatting y tipado estático.
  - Versionamiento de código con Git.

## Estructura del Proyecto

El proyecto está organizado en las siguientes carpetas y archivos:

- `.github/`: Configuración y workflows para integración continua (CI/CD).
- `data/`: Almacena los datos utilizados en el proyecto.
  - `external/`: Datos obtenidos de fuentes externas sin procesar.
  - `interim/`: Datos intermedios, transformados parcialmente.
  - `processed/`: Datos listos para ser usados en modelos de machine learning.
  - `raw/`: Datos originales sin modificar.
- `docs/`: Documentación y reportes del proyecto.
- `models/`: Almacenamiento de modelos entrenados y resultados.
- `notebooks/`: Jupyter notebooks para exploración, análisis y desarrollo del modelo.
- `venv/`: Entorno virtual con las dependencias del proyecto.
- `.gitignore`: Archivos y carpetas que no deben ser versionadas en Git.
- `README.md`: Descripción del proyecto, estructura y objetivos.
- `requirements.txt`: Lista de dependencias necesarias para reproducir el entorno.


### Descripción de los Datos

El dataset utilizado proviene del conjunto de datos **Telco Customer Churn** de IBM. Contiene información de clientes con características como:

- **Churn**: Indica si el cliente abandonó el servicio.
- **Servicios contratados**: Telefonía, internet, soporte técnico, seguridad, entre otros.
- **Información de cuenta**: Antigüedad, tipo de contrato, método de pago, facturación, cargos mensuales y totales.
- **Demografía**: Género, rango de edad, estado civil y dependientes.

Fuente original del dataset: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
Versión actualizada de IBM: [IBM Community](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

---
Este README proporciona una visión clara de la estructura y el propósito del proyecto, facilitando su colaboración y desarrollo.

## Configuración del Entorno  

Este proyecto utiliza **Python 3.12.9**. Para instalar las dependencias necesarias, ejecuta el siguiente comando en la terminal dentro del entorno virtual:  

```bash
pip install -r requirements.txt
