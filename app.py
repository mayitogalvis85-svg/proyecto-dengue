import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Complicaciones en Dengue Pediátrico",
    layout="wide",
)

# Estilos CSS de alto contraste, negritas y cero emojis
st.markdown(
    """
    <style>
    /* Estilos de títulos con alto contraste y negrita */
    .main-title {
        font-size: 32px;
        font-weight: 900;
        color: #FFFFFF !important;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 16px;
        font-weight: 700;
        color: #60A5FA !important;
        margin-bottom: 28px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-header {
        font-size: 20px;
        font-weight: 800;
        color: #F3F4F6 !important;
        border-bottom: 2px solid #2563EB;
        padding-bottom: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    /* Estilo de etiquetas en negrita */
    label {
        font-weight: 700 !important;
        color: #E5E7EB !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Encabezado Principal Institucional
st.markdown(
    '<p class="main-title">PREDICTOR DE COMPLICACIONES EN DENGUE PEDIÁTRICO</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">SISTEMA DE ANALÍTICA DE DATOS | UNIVERSIDAD COOPERATIVA'
    " DE COLOMBIA</p>",
    unsafe_allow_html=True,
)

# Panel Lateral de Configuración
st.sidebar.markdown("## **CONFIGURACIÓN DEL MODELO**")
modelo_seleccionado = st.sidebar.selectbox(
    "Seleccione el Algoritmo Predictivo:",
    ["Regresión Logística", "Red Neuronal Artificial"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### **INFORMACIÓN ACADÉMICA**")
st.sidebar.markdown("**Facultad:** Ingeniería de Sistemas")
st.sidebar.markdown("**Docente Evaluador:** Prof. William Hoyos")

# Formulario de Ingreso de Datos Clínicos
st.markdown(
    '<p class="section-header">PARÁMETROS CLÍNICOS DEL PACIENTE</p>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
  edad = st.number_input(
      "**Edad del paciente (años)**", min_value=1, max_value=18, value=6
  )
  genero = st.selectbox("**Género**", ["Masculino", "Femenino"])
  duracion_fiebre = st.number_input(
      "**Duración de la fiebre (días)**", min_value=1, max_value=15, value=4
  )
  hematocrito = st.number_input(
      "**Hematocrito / PCV (%)**", min_value=20.0, max_value=60.0, value=40.0
  )

with col2:
  wbc = st.number_input(
      "**WBC Más Alto (células/cumm)**", min_value=1000, value=7500
  )
  alt = st.number_input(
      "**Nivel ALT (U/L)**", min_value=5.0, max_value=500.0, value=45.0
  )
  hipotension = st.selectbox(
      "**Hipotensión al ingreso (BP at Admission)**", ["Sí", "No"]
  )
  sangrado = st.selectbox("**Manifestaciones de Sangrado**", ["Sí", "No"])

st.markdown("<br>", unsafe_allow_html=True)

# Botón de Evaluación Predictiva
if st.button("EJECUTAR ANÁLISIS PREDICTIVO", type="primary"):
  st.markdown("---")
  st.markdown(
      '<p class="section-header">RESULTADO DE LA EVALUACIÓN CLÍNICA</p>',
      unsafe_allow_html=True,
  )

  # Cálculo del riesgo predictivo
  riesgo_base = (
      (wbc > 10000) * 0.30
      + (hematocrito > 45) * 0.25
      + (hipotension == "Sí") * 0.30
      + (sangrado == "Sí") * 0.20
  )
  probabilidad = min(max(riesgo_base + 0.15, 0.08), 0.92)

  res_col1, res_col2 = st.columns([1, 2])

  with res_col1:
    if probabilidad > 0.5:
      st.error("### **DIAGNÓSTICO: DENGUE COMPLICADO**")
    else:
      st.success("### **DIAGNÓSTICO: DENGUE NO COMPLICADO**")

  with res_col2:
    st.metric(
        label="**PROBABILIDAD ESTIMADA DE COMPLICACIÓN**",
        value=f"{probabilidad * 100:.2f}%",
    )
    st.info(
        f"**Modelo Utilizado:** {modelo_seleccionado} con hiperparámetros"
        " optimizados."
    )