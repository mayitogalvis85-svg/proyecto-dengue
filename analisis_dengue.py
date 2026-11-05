import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 1. Cargar el archivo Excel
archivo_excel = 'MASTER_CHART_F1000Research.xlsx'
xls = pd.ExcelFile(archivo_excel)
df = pd.read_excel(archivo_excel, sheet_name=xls.sheet_names[0])

# Limpiar espacios en los nombres de las columnas
df.columns = [c.strip() for c in df.columns]

print("--- 1. DATOS CARGADOS CON ÉXITO ---")
print(f"Total de registros: {df.shape[0]}, Columnas: {df.shape[1]}")

# 2. Preparar la variable objetivo (Target) y las características (Features)
# Según el README: 1 = Dengue simple, 2 = Dengue complicado. Lo convertimos a 0 y 1.
y = (df['GROUPS: DENGUE FEVER OR COMPLICATED DENGUE'] == 2).astype(int)

# Seleccionamos las columnas numéricas clínicas y de laboratorio relevantes
features = ['AGE', 'GENDER', 'DURATION OF FEVER', 'HEADACHE', 'MYALGIA', 'ABDOMINAL PAIN', 
            'RASH', 'VOMITING', 'BREATHLESSNESS', 'BLEEDING', 'BP AT ADMISSION', 'ORGANOMEGALY', 
            'HB', 'PCV', 'LOWEST WBC', 'HIGHEST WBC', 'PLATELET COUNT', 'ALT', 'CREATININE', 'FERRITIN']

X = df[features]

# Rellenar valores faltantes (NaN) con la mediana de cada columna
X = X.fillna(X.median(numeric_only=True))

# 3. Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Estandarizar las variables (escalar para que tengan media 0 y desviación 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- 2. ENTRENANDO MODELO 1: REGRESIÓN LOGÍSTICA ---")
modelo_lr = LogisticRegression(random_state=42)
modelo_lr.fit(X_train_scaled, y_train)
y_pred_lr = modelo_lr.predict(X_test_scaled)

print(f"Precisión (Accuracy) - Regresión Logística: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC - Regresión Logística: {roc_auc_score(y_test, modelo_lr.predict_proba(X_test_scaled)[:, 1]):.4f}")

print("\n--- 3. ENTRENANDO MODELO 2: RED NEURONAL ARTIFICIAL ---")
modelo_nn = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
modelo_nn.fit(X_train_scaled, y_train)
y_pred_nn = modelo_nn.predict(X_test_scaled)

print(f"Precisión (Accuracy) - Red Neuronal: {accuracy_score(y_test, y_pred_nn):.4f}")
print(f"ROC-AUC - Red Neuronal: {roc_auc_score(y_test, modelo_nn.predict_proba(X_test_scaled)[:, 1]):.4f}")
print("\n¡Modelos entrenados y evaluados con éxito!")