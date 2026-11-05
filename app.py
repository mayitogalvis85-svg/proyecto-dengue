import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Configuración de la página web
st.set_page_config(page_title="Predictor de Complicaciones en Dengue Pediátrico", page_icon="🏥", layout="centered")

st.title("🏥 Predictor de Complicaciones en Dengue Pediátrico")
st.write("Proyecto Final - Analítica de Datos | Universidad Cooperativa de Colombia")
st.markdown("---")

# Cargar y entrenar los modelos automáticamente usando el dataset
@st.cache_resource
def cargar_y_entrenar():
    archivo_excel = 'MASTER_CHART_F1000Research.xlsx'
    xls = pd.ExcelFile(archivo_excel)
    df = pd.read_excel(archivo_excel, sheet_name=xls.sheet_names[0])
    df.columns = [c.strip() for c in df.columns]
    
    y = (df['GROUPS: DENGUE FEVER OR COMPLICATED DENGUE'] == 2).astype(int)
    
    features = ['AGE', 'GENDER', 'DURATION OF FEVER', 'HEADACHE', 'MYALGIA', 'ABDOMINAL PAIN', 
                'RASH', 'VOMITING', 'BREATHLESSNESS', 'BLEEDING', 'BP AT ADMISSION', 'ORGANOMEGALY', 
                'HB', 'PCV', 'LOWEST WBC', 'HIGHEST WBC', 'PLATELET COUNT', 'ALT', 'CREATININE', 'FERRITIN']
    
    X = df[features].fillna(df[features].median(numeric_only=True))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar modelos
    lr = LogisticRegression(random_state=42).fit(X_scaled, y)
    nn = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42).fit(X_scaled, y)
    
    return lr, nn, scaler, X.median().tolist()

lr_model, nn_model, scaler, valores_medianos = cargar_y_entrenar()

# Selector de Modelo en la interfaz
st.sidebar.header("Configuración")
modelo_elegido = st.sidebar.selectbox("Seleccione el Modelo de Analítica:", ["Regresión Logística", "Red Neuronal Artificial"])

st.subheader("📝 Ingrese los Parámetros Clínicos del Paciente:")

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad (años)", min_value=0, max_value=18, value=6)
    genero = st.selectbox("Género", options=[1, 2], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
    duracion_fiebre = st.number_input("Duración de fiebre (días)", min_value=1, max_value=15, value=4)
    pcv = st.number_input("Hematocrito / PCV (%)", min_value=20.0, max_value=60.0, value=40.0)
    plaquetas = st.number_input("Conteo de Plaquetas (cél/cumm)", min_value=5000, max_value=500000, value=70000)

with col2:
    wbc = st.number_input("WBC Más Alto (cél/cumm)", min_value=1000, max_value=30000, value=7500)
    alt = st.number_input("ALT (U/L)", min_value=5.0, max_value=1000.0, value=45.0)
    hipotension = st.selectbox("Hipotensión (BP at Admission)", options=[1, 2], format_func=lambda x: "Sí" if x == 1 else "No")
    sangrado = st.selectbox("Manifestaciones de Sangrado", options=[1, 2], format_func=lambda x: "Sí" if x == 1 else "No")
    organomegalia = st.selectbox("Organomegalia", options=[1, 2], format_func=lambda x: "Sí" if x == 1 else "No")

# Botón para realizar la predicción
if st.button("🔍 Predecir Riesgo de Complicación", type="primary"):
    # Tomamos los valores medianos por defecto y reemplazamos con los que ingresó el usuario
    datos_usuario = valores_medianos.copy()
    datos_usuario[0] = edad
    datos_usuario[1] = genero
    datos_usuario[2] = duracion_fiebre
    datos_usuario[10] = hipotension
    datos_usuario[11] = organomegalia
    datos_usuario[13] = pcv
    datos_usuario[15] = wbc
    datos_usuario[16] = plaquetas
    datos_usuario[17] = alt
    
    # Escalar datos de entrada
    entrada_escalada = scaler.transform([datos_usuario])
    
    # Predicción según el modelo seleccionado
    if modelo_elegido == "Regresión Logística":
        prediccion = lr_model.predict(entrada_escalada)[0]
        probabilidad = lr_model.predict_proba(entrada_escalada)[0][1]
    else:
        prediccion = nn_model.predict(entrada_escalada)[0]
        probabilidad = nn_model.predict_proba(entrada_escalada)[0][1]
        
    st.markdown("---")
    st.subheader("📊 Resultado del Análisis:")
    
    if prediccion == 1:
        st.error(f"⚠️ **Alerta:** El modelo predice **Dengue Complicado** (Probabilidad de riesgo: {probabilidad:.2%})")
        st.markdown("*(Se sugiere hospitalización para observación y manejo clínico estrecho).*")
    else:
        st.success(f"✅ **Resultado:** El modelo predice **Dengue No Complicado** (Probabilidad de complicación: {probabilidad:.2%})")
        st.markdown("*(El paciente puede mantenerse bajo manejo ambulatorio controlado).*")