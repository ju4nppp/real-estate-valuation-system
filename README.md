# Neojaus AI Valuation System 

Este proyecto implementa un sistema de Machine Learning para la valuación automática y clasificación de precios de propiedades inmobiliarias en México.

El sistema utiliza **LightGBM** para estimar el valor comercial de una propiedad y aplica lógica de negocio estadística para detectar oportunidades de mercado (propiedades subvaluadas o sobrevaluadas).

## Demo Interactiva
Prueba el modelo en tiempo real aquí: https://huggingface.co/spaces/ju4nppp/Mexico_housing_price_classifier

## Enfoque del Proyecto

El proyecto evolucionó de un enfoque tradicional de clasificación a un sistema híbrido más robusto:

1.  **Desafío:** Escalar de un dataset local a uno nacional (+100k registros) con alta cardinalidad (+33,000 colonias).
2.  **Solución:**
    * **Motor:** Se migró de Random Forest a **LightGBM** por eficiencia y manejo nativo de categorías.
    * **Estrategia:** Se descubrió que un modelo de regresión único de alta precisión ($R^2 > 0.85$) combinado con **umbrales de percentiles dinámicos** (-19% / +16%) era superior a intentar clasificar errores con un segundo modelo.

## Stack Tecnológico

* **Python 3.11**
* **LightGBM:** Algoritmo principal (Regresión y Clasificación).
* **Scikit-Learn:** Preprocesamiento y Pipelines.
* **Pandas/Numpy:** Manipulación de datos.
* **Gradio:** Interfaz web para demostración.
* **Joblib:** Serialización de modelos.

## Estructura del Repositorio

* `training_v2.py`: Pipeline completo de ETL y entrenamiento. Genera el modelo y las reglas de negocio.
* `inference_v2.py`: Script de producción. Carga los artefactos y expone la función `get_property_label()`. Incluye validaciones de "Zonas Confiables".
* `app.py`: Aplicación frontend para Hugging Face Spaces.
* `explore_new_data.ipynb`: Notebook de Análisis Exploratorio de Datos (EDA) y justificación de features.

## Resultados

* **R² Score (Regresión):** 0.852
* **Feature Importance:** Las variables dominantes fueron `superficieConstruccionM2`, `latitud`, `longitud` y `colonia`.

## Instalación y Uso

1.  Clonar el repositorio:
    ```bash
    git clone [https://github.com/ju4nppp/real-estate-valuation-system.git](https://github.com/ju4nppp/real-estate-valuation-system.git)
    ```
    
2.  Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```
    
3.  Ejecutar inferencia de prueba:
    ```bash
    python inference_v2.py
    ```

---
*Desarrollado durante prácticas profesionales para Neojaus.*
