from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = Path(__file__).resolve().parents[1] / "MASTER_CHART_F1000Research.xlsx"

FEATURE_ORDER = [
    "age_group",
    "gender",
    "fever_duration",
    "headache",
    "myalgia",
    "abdominal_pain",
    "rash",
    "vomiting",
    "breathlessness",
    "bleeding",
    "hypotension",
    "organomegaly",
    "hb",
    "pcv",
    "wbc_min",
    "wbc_max",
    "platelets",
    "alt",
    "creatinine",
    "ferritin",
]
FEATURE_LABELS = {
    "pcv": "PCV (%)",
    "platelets": "Plaquetas (células/cumm)",
    "wbc_max": "WBC máximo (células/cumm)",
    "alt": "ALT (U/L)",
    "hypotension": "Hipotensión a la admisión",
    "age_group": "Edad (grupo)",
    "gender": "Género",
    "fever_duration": "Duración de la fiebre (días)",
    "headache": "Dolor de cabeza",
    "myalgia": "Mialgia",
    "abdominal_pain": "Dolor abdominal",
    "rash": "Erupción cutánea",
    "vomiting": "Vómito",
    "breathlessness": "Dificultad respiratoria",
    "bleeding": "Manifestaciones de sangrado",
    "organomegaly": "Organomegalia",
    "hb": "Hemoglobina (g/dL)",
    "wbc_min": "WBC mínimo (células/cumm)",
    "creatinine": "Creatinina (mg/dL)",
    "ferritin": "Ferritina (ng/mL)",
}

BINARY_MAPS = {
    "age_group": {1: 0, 2: 1},
    "gender": {1: 0, 2: 1},
    "headache": {1: 1, 2: 0},
    "myalgia": {1: 1, 2: 0},
    "abdominal_pain": {1: 1, 2: 0},
    "rash": {1: 1, 2: 0},
    "vomiting": {1: 1, 2: 0},
    "breathlessness": {1: 1, 2: 0},
    "bleeding": {1: 1, 2: 0},
    "hypotension": {1: 1, 2: 0},
    "organomegaly": {1: 1, 2: 0},
}


def normalize_column_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _normalized_columns(data: pd.DataFrame) -> dict[str, str]:
    return {normalize_column_name(column): column for column in data.columns}


def _find_column(data: pd.DataFrame, aliases: list[str]) -> str | None:
    columns = _normalized_columns(data)
    for alias in aliases:
        normalized = normalize_column_name(alias)
        if normalized in columns:
            return columns[normalized]
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _decode_target(series: pd.Series) -> pd.Series:
    numeric = _to_numeric(series)
    values = set(numeric.dropna().unique())
    if values and values.issubset({1, 2}) and 2 in values:
        return numeric.map({1: 0, 2: 1})
    return numeric.where(numeric.isin([0, 1]))


def _decode_binary(series: pd.Series, mapping: dict[int, int]) -> pd.Series:
    numeric = _to_numeric(series)
    values = set(numeric.dropna().unique())
    if values.issubset({0, 1}):
        return numeric
    return numeric.map(mapping).fillna(numeric)


def load_source_data(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")
    return pd.read_excel(path)


def clean_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    source = data.copy()
    original_rows = len(source)
    source.columns = [normalize_column_name(column) for column in source.columns]

    required = {
        "target": ["groups_dengue_fever_or_complicated_dengue", "groups", "target"],
        "age_group": ["age", "age_group"],
        "gender": ["gender", "sex"],
        "fever_duration": ["duration_of_fever", "fever_duration"],
        "headache": ["headache"],
        "myalgia": ["myalgia"],
        "abdominal_pain": ["abdominal_pain"],
        "rash": ["rash"],
        "vomiting": ["vomiting"],
        "breathlessness": ["breathlessness"],
        "bleeding": ["bleeding", "bleeding_manifestations"],
        "hypotension": ["bp_at_admission", "hypotension", "blood_pressure"],
        "organomegaly": ["organomegaly"],
        "hb": ["hb", "haemoglobin", "hemoglobin"],
        "pcv": ["pcv"],
        "wbc_min": ["lowest_wbc", "wbc_min"],
        "wbc_max": ["highest_wbc", "wbc_max"],
        "platelets": ["platelet_count", "platelets", "platelet"],
        "alt": ["alt", "alanine_aminotransferase"],
        "creatinine": ["creatinine"],
        "ferritin": ["ferritin"],
    }
    selected: dict[str, str] = {}
    for feature, aliases in required.items():
        match = _find_column(source, aliases)
        if match is None:
            raise ValueError(f"No se encontró la columna requerida para '{feature}'.")
        selected[feature] = match

    cleaned = pd.DataFrame(index=source.index)
    for feature in FEATURE_ORDER:
        cleaned[feature] = _to_numeric(source[selected[feature]])

    target = _decode_target(source[selected["target"]])
    valid = target.notna()
    cleaned = cleaned.loc[valid].copy()
    target = target.loc[valid].astype(int)

    for feature, mapping in BINARY_MAPS.items():
        cleaned[feature] = _decode_binary(cleaned[feature], mapping)
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    duplicate_mask = cleaned.duplicated(keep="first")
    cleaned = cleaned.loc[~duplicate_mask].copy()
    target = target.loc[cleaned.index]

    stats = {
        "original_rows": original_rows,
        "rows_after_cleaning": len(cleaned),
        "duplicates_removed": int(duplicate_mask.sum()),
        "missing_alt": int(cleaned["alt"].isna().sum()),
        "missing_ferritin": int(cleaned["ferritin"].isna().sum()),
    }
    return cleaned[FEATURE_ORDER], target, stats


def _target_column(data: pd.DataFrame) -> str | None:
    return _find_column(
        data,
        [
            "target",
            "target_opcional",
            "groups",
            "groups_dengue_fever_or_complicated_dengue",
            "dengue_group",
            "complicated_dengue",
        ],
    )


def prepare_batch(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    if data.empty:
        raise ValueError("El archivo cargado no contiene registros.")

    columns = {
        "age_group": ["age", "age_group"],
        "gender": ["gender", "sex"],
        "fever_duration": ["duration_of_fever", "fever_duration"],
        "headache": ["headache"],
        "myalgia": ["myalgia"],
        "abdominal_pain": ["abdominal_pain"],
        "rash": ["rash"],
        "vomiting": ["vomiting"],
        "breathlessness": ["breathlessness"],
        "bleeding": ["bleeding", "bleeding_manifestations"],
        "hypotension": ["bp_at_admission", "hypotension", "blood_pressure"],
        "organomegaly": ["organomegaly"],
        "hb": ["hb", "haemoglobin", "hemoglobin"],
        "pcv": ["pcv"],
        "wbc_min": ["lowest_wbc", "wbc_min"],
        "wbc_max": ["highest_wbc", "wbc_max"],
        "platelets": ["platelet_count", "platelets", "platelet"],
        "alt": ["alt", "alanine_aminotransferase"],
        "creatinine": ["creatinine"],
        "ferritin": ["ferritin"],
    }
    features = pd.DataFrame(index=data.index)
    for feature, aliases in columns.items():
        match = _find_column(data, aliases)
        if match is None:
            raise ValueError(
                "Faltan columnas requeridas. Se necesitan PCV, plaquetas, WBC máximo, ALT y presión arterial."
            )
        features[feature] = _to_numeric(data[match])

    for feature, mapping in BINARY_MAPS.items():
        features[feature] = _decode_binary(features[feature], mapping)
    features = features[FEATURE_ORDER].replace([np.inf, -np.inf], np.nan)
    if features.isna().all(axis=1).any():
        raise ValueError("Se encontró al menos una fila sin ningún valor numérico válido.")

    target_match = _target_column(data)
    target = None
    if target_match is not None:
        target = _decode_target(data[target_match])
        if target.isna().any():
            raise ValueError("La variable objetivo contiene valores que no se pueden interpretar como 0/1 o 1/2.")
        target = target.astype(int)
    return features, target


def score_patient(values: dict[str, float]) -> tuple[int, str]:
    score = 0
    score += int(values["pcv"] >= 42)
    score += int(values["platelets"] <= 75_000)
    score += int(values["wbc_max"] >= 7_000)
    score += int(values["alt"] >= 70)
    score += 3 * int(values["hypotension"] == 1)
    label = "Mayor riesgo" if score >= 2 else "Menor riesgo"
    return score, label
