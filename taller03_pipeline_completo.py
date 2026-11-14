"""
Taller 03 - Reto Integrador de Analítica Predictiva: Carreras de Atletismo
Pipeline Completo End-to-End (Fases 1 a 4):
- Auditoría de calidad y reglas de dominio
- División temporal estricta por bloque de carrera (Train / Validation / Test)
- Imputación sin fuga de información (sin data leakage)
- Línea base vs. Regresión Lineal Aditiva vs. Regresión Lineal con Interacciones Físicas
- Evaluación global, segmentada y empaquetado del Pipeline en joblib
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    median_absolute_error,
)

MEDIAN_PACE_TRAIN = 5.6575


def aplicar_reglas_dominio_y_features(df_in: pd.DataFrame, pace_fill: float = MEDIAN_PACE_TRAIN) -> pd.DataFrame:
    """
    Aplica reglas de negocio deportivas (convierte valores imposibles a NaN para imputación)
    y construye términos de interacción física distancia_km * predictores.
    """
    d = df_in.copy()
    # Reglas de dominio sobre entrenamiento semanal
    d.loc[(d["km_entrenamiento_semana"] <= 0) | (d["km_entrenamiento_semana"] > 160), "km_entrenamiento_semana"] = np.nan
    d.loc[(d["sesiones_semana"] <= 0) | (d["sesiones_semana"] > 14), "sesiones_semana"] = np.nan
    d.loc[(d["tirada_larga_km"] <= 0) | (d["tirada_larga_km"] > 50), "tirada_larga_km"] = np.nan
    d.loc[d["tirada_larga_km"] > d["km_entrenamiento_semana"], "tirada_larga_km"] = np.nan

    # Indicador explícito de debutante (ausencia estructural en ritmo_promedio_previo_min_km)
    d["es_debutante"] = (d["carreras_previas"] == 0).astype(int)
    pace_filled = d["ritmo_promedio_previo_min_km"].fillna(pace_fill)

    # Interacciones físicas (Tiempo = Distancia x Ritmo efectivo)
    d["dist_x_ritmo_prev"] = d["distancia_km"] * pace_filled
    d["dist_x_desnivel"] = d["distancia_km"] * d["desnivel_positivo_m"]
    d["dist_x_debutante"] = d["distancia_km"] * d["es_debutante"]
    d["dist_x_km_sem"] = d["distancia_km"] * d["km_entrenamiento_semana"]
    d["dist_x_exp"] = d["distancia_km"] * d["anios_experiencia"]
    d["dist_x_edad"] = d["distancia_km"] * d["edad_en_carrera"]
    d["dist_x_temp"] = d["distancia_km"] * d["temperatura_c"]
    d["dist_x_trail"] = d["distancia_km"] * (d["tipo_carrera"] == "Trail").astype(int)
    d["dist_x_montana"] = d["distancia_km"] * (d["tipo_carrera"] == "Montaña").astype(int)
    return d


def evaluar_metricas(y_true, y_pred) -> dict:
    return {
        "MAE_min": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE_min": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "MedAE_min": round(float(median_absolute_error(y_true, y_pred)), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAPE_%": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 4),
    }


def main():
    df = pd.read_excel("Dataset_Carreras_Atletismo_12000.xlsx", sheet_name="Participaciones")
    df["fecha_carrera"] = pd.to_datetime(df["fecha_carrera"])

    # División temporal obligatoria por bloque íntegro de carrera
    carreras_ord = (
        df[["id_carrera", "fecha_carrera"]]
        .drop_duplicates()
        .sort_values("fecha_carrera")
        .reset_index(drop=True)
    )
    train_ids = set(carreras_ord.loc[:33, "id_carrera"])   # 34 carreras (8,500 reg)
    val_ids = set(carreras_ord.loc[34:40, "id_carrera"])   # 7 carreras  (1,750 reg)
    test_ids = set(carreras_ord.loc[41:, "id_carrera"])    # 7 carreras  (1,750 reg)

    train_raw = df[df["id_carrera"].isin(train_ids)].copy()
    val_raw = df[df["id_carrera"].isin(val_ids)].copy()
    test_raw = df[df["id_carrera"].isin(test_ids)].copy()

    pace_fill_train = float(train_raw["ritmo_promedio_previo_min_km"].median())
    train_df = aplicar_reglas_dominio_y_features(train_raw, pace_fill_train)
    val_df = aplicar_reglas_dominio_y_features(val_raw, pace_fill_train)
    test_df = aplicar_reglas_dominio_y_features(test_raw, pace_fill_train)

    num_cols = [
        "distancia_km", "desnivel_positivo_m", "edad_en_carrera",
        "km_entrenamiento_semana", "sesiones_semana", "tirada_larga_km",
        "anios_experiencia", "temperatura_c", "ritmo_promedio_previo_min_km",
        "altitud_salida_msnm", "dias_descanso_semana", "carreras_previas", "es_debutante",
        "dist_x_ritmo_prev", "dist_x_desnivel", "dist_x_debutante",
        "dist_x_km_sem", "dist_x_exp", "dist_x_edad", "dist_x_temp",
        "dist_x_trail", "dist_x_montana",
    ]
    cat_cols = ["tipo_carrera", "sexo"]

    preprocesador = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ])

    pipeline = Pipeline([
        ("prep", preprocesador),
        ("reg", LinearRegression()),
    ])

    pipeline.fit(train_df[num_cols + cat_cols], train_df["tiempo_llegada_min"])

    print("Métricas Validación:", evaluar_metricas(val_df["tiempo_llegada_min"], pipeline.predict(val_df[num_cols + cat_cols])))
    print("Métricas Test Final:", evaluar_metricas(test_df["tiempo_llegada_min"], pipeline.predict(test_df[num_cols + cat_cols])))

    joblib.dump(pipeline, "modelo_atletismo_pipeline.joblib")
    print("Pipeline guardado exitosamente en modelo_atletismo_pipeline.joblib")


if __name__ == "__main__":
    main()
