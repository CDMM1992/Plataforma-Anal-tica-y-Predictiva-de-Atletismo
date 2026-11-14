"""
Fase 5 — Implementación Web End-to-End con Streamlit y Plotly
Tablero Descriptivo + Tablero Predictivo de Carreras de Atletismo
Ejecución: streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from taller03_pipeline_completo import aplicar_reglas_dominio_y_features, MEDIAN_PACE_TRAIN

st.set_page_config(page_title="Analítica Predictiva - Atletismo", layout="wide", page_icon="🏃")

@st.cache_data
def cargar_datos():
    df = pd.read_excel("Dataset_Carreras_Atletismo_12000.xlsx", sheet_name="Participaciones")
    df["fecha_carrera"] = pd.to_datetime(df["fecha_carrera"])
    df["anio"] = df["fecha_carrera"].dt.year
    # Depuración para capa descriptiva
    df.loc[(df["km_entrenamiento_semana"] <= 0) | (df["km_entrenamiento_semana"] > 160), "km_entrenamiento_semana"] = np.nan
    df.loc[(df["sesiones_semana"] <= 0) | (df["sesiones_semana"] > 14), "sesiones_semana"] = np.nan
    df.loc[(df["tirada_larga_km"] <= 0) | (df["tirada_larga_km"] > 50), "tirada_larga_km"] = np.nan
    return df

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_atletismo_pipeline.joblib")

df = cargar_datos()
modelo = cargar_modelo()

st.title("🏃 Plataforma Analítica y Predictiva de Carreras de Atletismo")
tab_desc, tab_pred = st.tabs(["📊 Tablero Descriptivo", "🎯 Simulador Predictivo"])

with tab_desc:
    st.sidebar.header("Filtros Cruzados")
    anios = st.sidebar.multiselect("Año", sorted(df["anio"].unique()), default=sorted(df["anio"].unique()))
    ciudades = st.sidebar.multiselect("Ciudad Carrera", sorted(df["ciudad_carrera"].unique()), default=sorted(df["ciudad_carrera"].unique()))
    tipos = st.sidebar.multiselect("Tipo de Carrera", sorted(df["tipo_carrera"].unique()), default=sorted(df["tipo_carrera"].unique()))
    distancias = st.sidebar.multiselect("Distancia (km)", sorted(df["distancia_km"].unique()), default=sorted(df["distancia_km"].unique()))
    sexos = st.sidebar.multiselect("Sexo", sorted(df["sexo"].unique()), default=sorted(df["sexo"].unique()))

    dff = df[
        df["anio"].isin(anios) &
        df["ciudad_carrera"].isin(ciudades) &
        df["tipo_carrera"].isin(tipos) &
        df["distancia_km"].isin(distancias) &
        df["sexo"].isin(sexos)
    ]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Participaciones", f"{len(dff):,}")
    c2.metric("Tiempo Promedio", f"{dff['tiempo_llegada_min'].mean():.1f} min")
    c3.metric("Ritmo Promedio", f"{dff['ritmo_real_min_km'].mean():.2f} min/km")
    c4.metric("Volumen Semanal", f"{dff['km_entrenamiento_semana'].mean():.1f} km")
    prob_podio = (dff["podio_general"] == "Sí").mean() * 100 if len(dff) > 0 else 0
    c5.metric("Tasa de Podio", f"{prob_podio:.2f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.box(dff, x="distancia_km", y="tiempo_llegada_min", color="tipo_carrera",
                      title="Distribución de Tiempos por Distancia y Modalidad")
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        agg_city = dff.groupby(["ciudad_carrera", "tipo_carrera"], as_index=False)["ritmo_real_min_km"].mean()
        fig2 = px.bar(agg_city, x="ciudad_carrera", y="ritmo_real_min_km", color="tipo_carrera", barmode="group",
                      title="Ritmo Promedio (min/km) por Ciudad y Modalidad")
        st.plotly_chart(fig2, use_container_width=True)

with tab_pred:
    st.subheader("Estimación Individual de Tiempo de Llegada")
    col1, col2, col3 = st.columns(3)
    with col1:
        distancia_km = st.selectbox("Distancia (km)", [5.0, 10.0, 15.0, 21.1, 30.0, 42.195], index=3)
        tipo_carrera = st.selectbox("Tipo de Carrera", ["Ruta", "Trail", "Montaña"])
        desnivel_m = st.number_input("Desnivel Positivo (m)", min_value=0, max_value=4000, value=250)
        altitud_msnm = st.number_input("Altitud Salida (msnm)", min_value=0, max_value=3500, value=1480)
        temperatura_c = st.slider("Temperatura (°C)", 4.0, 32.0, 18.5)
    with col2:
        sexo = st.radio("Sexo", ["M", "F"], horizontal=True)
        edad = st.slider("Edad en Carrera", 16, 70, 32)
        experiencia = st.slider("Años de Experiencia", 0, 25, 5)
        carreras_prev = st.number_input("Carreras Previas", min_value=0, max_value=50, value=3)
        es_debut = carreras_prev == 0
        ritmo_prev = st.number_input("Ritmo Promedio Previo (min/km)", min_value=2.8, max_value=9.0, value=5.20, disabled=es_debut)
    with col3:
        km_sem = st.slider("Km Entrenamiento / Semana", 5.0, 140.0, 42.0)
        sesiones_sem = st.slider("Sesiones / Semana", 1, 12, 5)
        tirada_larga = st.slider("Tirada Larga (km)", 2.0, min(50.0, km_sem), min(18.0, km_sem))
        dias_desc = st.slider("Días de Descanso / Semana", 0, 5, 2)

    input_df = pd.DataFrame([{
        "distancia_km": distancia_km,
        "desnivel_positivo_m": desnivel_m,
        "edad_en_carrera": edad,
        "km_entrenamiento_semana": km_sem,
        "sesiones_semana": sesiones_sem,
        "tirada_larga_km": tirada_larga,
        "anios_experiencia": experiencia,
        "temperatura_c": temperatura_c,
        "ritmo_promedio_previo_min_km": np.nan if es_debut else ritmo_prev,
        "altitud_salida_msnm": altitud_msnm,
        "dias_descanso_semana": dias_desc,
        "carreras_previas": carreras_prev,
        "tipo_carrera": tipo_carrera,
        "sexo": sexo,
    }])

    feat_df = aplicar_reglas_dominio_y_features(input_df, MEDIAN_PACE_TRAIN)
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
    pred_min = float(modelo.predict(feat_df[num_cols + cat_cols])[0])
    ritmo_est = pred_min / distancia_km
    horas = int(pred_min // 60)
    mins = int(pred_min % 60)
    segs = int(round((pred_min - int(pred_min)) * 60))

    st.success(f"⏱️ Tiempo Estimado: **{pred_min:.2f} min** ({horas:02d}:{mins:02d}:{segs:02d}) | Ritmo Proyectado: **{ritmo_est:.2f} min/km**")
