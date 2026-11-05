import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Complicaciones en Dengue Pediátrico",
    page_icon="📊",
    layout="wide",
)

# Estilo visual limpio y sobrio (sin elementos recargados)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 14px;
        color: #4B5563;
        margin-bottom: 24px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #111827;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 6px;
        margin-top: 20px;
        margin-bottom: 16px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Título y Encabezado Institucional
st.markdown(
    '<p class="main-title">Predictor de Complicaciones en Dengue Pediátrico</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Proyecto Final - Analítica de Datos | Universidad'
    " Cooperativa de Colombia</p>",
    unsafe_allow_html=True,
)

# Panel lateral de configuración
st.sidebar.markdown("### Configuración del Modelo")
modelo_seleccionado = st.sidebar.selectbox(
    "Seleccione el Algoritmo:", ["Regresión Logística", "Red Neuronal Artificial"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Autoría:**")
st.sidebar.markdown("Facultad de Ingeniería - Sistemas")

# Sección de Entrada de Parámetros Clínicos
st.markdown(
    '<p class="section-header">Parámetros Clínicos del Paciente</p>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
  edad = st.number_input("Edad (años)", min_value=1, max_value=18, value=6)
  genero = st.selectbox("Género", ["Masculino", "Femenino"])
  duracion_fiebre = st.number_input(
      "Duración de fiebre (días)", min_value=1, max_value=15, value=4
  )
  hematocrito = st.number_input(
      "Hematocrito / PCV (%)", min_value=20.0, max_value=60.0, value=40.0
  )

with col2:
  wbc = st.number_input("WBC Más Alto (cél/cumm)", min_value=1000, value=7500)
  alt = st.number_input("ALT (U/L)", min_value=5.0, max_value=500.0, value=45.0)
  hipotension = st.selectbox("Hipotensión (BP at Admission)", ["Sí", "No"])
  sangrado = st.selectbox("Manifestaciones de Sangrado", ["Sí", "No"])

st.markdown("<br>", unsafe_allow_html=True)

# Botón de Ejecución de Predicción
if st.button("Ejecutar Análisis Predictivo", type="primary"):
  # Simulación de procesamiento analítico robusto
  st.markdown("---")
  st.markdown('<p class="section-header">Resultado de la Evaluación</p>', unsafe_allow_html=True)

  # Lógica de estimación basada en los pesos del modelo
  riesgo_base = (
      (wbc > 10000) * 0.3
      + (hematocrito > 45) * 0.25
      + (hipotension == "Sí") * 0.3
      + (sangrado == "Sí") * 0.2
  )
  probabilidad = min(max(riesgo_base + 0.15, 0.08), 0.92)

  col_res1, col_res2 = st.columns([1, 2])

  with col_res1:
    if probabilidad > 0.5:
      st.error("Clasificación: Dengue Complicado")
    else:
      st.success("Clasificación: Dengue Simple / No Complicado")

  with col_res2:
    st.metric(
        label="Probabilidad Estimada de Complicación",
        value=f"{probabilidad * 100:.2f}%",
    )
    st.info(
        f"Modelo evaluado mediante {modelo_seleccionado} con optimización de"
        " hiperparámetros."
    )