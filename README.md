# Vitalia Analytics

Aplicación académica para comparar regresión logística y una red neuronal artificial en la estimación del riesgo de complicaciones en casos pediátricos de dengue.

## Ejecución local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

En PowerShell, puede usarse el script de activación `.venv\Scripts\Activate.ps1`; en Linux o macOS usar `source .venv/bin/activate`.

## Notebook de análisis

El análisis principal está documentado en `notebooks/01_analisis_y_modelado.ipynb`. Allí se realizan la exploración, limpieza, entrenamiento de los dos modelos, evaluación y exportación de resultados.

## Arquitectura

- `notebooks/01_analisis_y_modelado.ipynb`: análisis, limpieza, entrenamiento y evaluación de los modelos.
- `outputs/modelos.joblib`: modelos entrenados y exportados desde el notebook.
- `app.py`: servidor web Flask que carga los modelos y expone las predicciones.
- `templates/` y `static/`: interfaz web de Vitalia Analytics.

## Despliegue en Render

- Crear un repositorio en GitHub con estos archivos y el dataset.
- Crear un Web Service en Render conectado al repositorio.
- Build command: `pip install -r requirements.txt`.
- Start command: `gunicorn app:app`.

El archivo `render.yaml` contiene esta configuración.

## Variables de entrada

- Edad y género.
- Duración de la fiebre.
- Dolor de cabeza, mialgia, dolor abdominal, erupción, vómito y dificultad respiratoria.
- Manifestaciones de sangrado, hipotensión y organomegalia.
- Hemoglobina, PCV, WBC mínimo, WBC máximo, plaquetas, ALT, creatinina y ferritina.

No se utilizan las variables de tratamiento (`IV fluids`, `antibiotics`, `antipyretics` y `steroids`) porque pueden ocurrir después de la evaluación inicial.

La aplicación acepta archivos CSV y Excel para predicción por lote. Para calcular métricas del archivo cargado debe incluirse una variable objetivo con valores `1/2` o `0/1`.

## Fuente y limitaciones

El dataset corresponde al artículo de Bhaskar et al. (2022), *Predictive scoring system for risk of complications in pediatric dengue infection*, DOI: https://doi.org/10.12688/f1000research.111214.1.

Vitalia Analytics es una demostración educativa. No es un dispositivo médico ni reemplaza la valoración de profesionales de la salud. El conjunto de datos es pequeño y procede de una población específica, por lo que los resultados no deben generalizarse automáticamente a otros contextos.
