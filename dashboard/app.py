"""
Dashboard de resultados 

Muestra:
- Visualizaciones de calidad del aire y tránsito.
- Análisis de correlaciones.
- Mapa de zonas críticas.
- Comparación de modelos de Machine Learning.
- Predicciones de PM2.5.

El dashboard utiliza los módulos desarrollados en:
    src/visualizacion/
    src/modelos/
    src/helpers/
"""

from __future__ import annotations

import sys
from pathlib import Path

# CONFIGURACIÓN DE RUTAS

# Ruta raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Ruta de src
SRC_DIR = ROOT_DIR / "src"

# Permitir importar los módulos de src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# IMPORTACIONES

import streamlit as st
import pandas as pd

from helpers.datos_demo import generar_datos_demo
from visualizacion.visualizador import Visualizador
from modelos.ModeloML import ModeloML

# CONFIGURACIÓN DEL DASHBOARD

st.set_page_config(
    page_title="Calidad del Aire y Tránsito",
    page_icon="🌎",
    layout="wide",
)

# TÍTULO

st.title("🌎 Dashboard de Calidad del Aire y Tránsito")

st.markdown(
    """
    Este dashboard presenta los resultados de análisis exploratorio,
    visualizaciones y modelos de Machine Learning relacionados con
    la calidad del aire y el flujo vehicular.
    """
)

# CARGA DE DATOS

@st.cache_data
def cargar_datos():
    """
    Carga los datos utilizados por el dashboard.

    Actualmente utiliza datos DEMO para comprobar que todos los
    módulos funcionan correctamente.

    Cuando Persona A/B entregue el DataFrame real, esta función
    puede sustituirse por la carga del dataset real.
    """
    return generar_datos_demo()


df = cargar_datos()

# INFORMACIÓN GENERAL DE LOS DATOS

st.header("📊 Datos utilizados")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Registros",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Columnas",
        f"{len(df.columns)}"
    )

with col3:
    st.metric(
        "Variable objetivo",
        "PM2.5"
    )


with st.expander("Ver datos"):
    st.dataframe(
        df.head(20),
        use_container_width=True
    )

# CREAR VISUALIZADOR

viz = Visualizador(df)

# SECCIÓN DE VISUALIZACIONES

st.header("📈 Visualizaciones")

# 1. Flujo vehicular vs PM2.5

st.subheader("Flujo vehicular vs PM2.5")

fig = viz.graficar_flujo_vs_pm25()

if fig is not None:
    st.plotly_chart(
        fig,
        use_container_width=True
    )
else:
    st.warning(
        "No hay suficientes datos para mostrar esta visualización."
    )

# 2. Flujo vehicular vs NO2

st.subheader("Flujo vehicular vs NO2")

fig = viz.graficar_flujo_vs_no2()

if fig is not None:
    st.plotly_chart(
        fig,
        use_container_width=True
    )
else:
    st.warning(
        "No hay suficientes datos para mostrar esta visualización."
    )

# 3. Series temporales

st.subheader("Serie temporal de contaminación")

fig = viz.graficar_series_temporales()

if fig is not None:
    st.plotly_chart(
        fig,
        use_container_width=True
    )
else:
    st.warning(
        "No hay datos suficientes para mostrar la serie temporal."
    )

# 4. PM2.5 vs meteorología

st.subheader("PM2.5 frente a variables meteorológicas")

fig = viz.graficar_pm25_vs_meteorologia()

if fig is not None:
    st.plotly_chart(
        fig,
        use_container_width=True
    )
else:
    st.warning(
        "No hay suficientes variables meteorológicas disponibles."
    )

# 5. Heatmap

st.subheader("Mapa de correlaciones")

fig = viz.graficar_heatmap_correlaciones()

if fig is not None:
    st.plotly_chart(
        fig,
        use_container_width=True
    )
else:
    st.warning(
        "No hay suficientes columnas numéricas para generar el heatmap."
    )

# 6. Mapa de zonas críticas

st.subheader("🗺️ Zonas críticas")

fig = viz.mapa_zonas_criticas(
    metrica="pm25",
    agregacion="mean"
)

if fig is not None:
    st.plotly_chart(
        fig,
        use_container_width=True
    )
else:
    st.warning(
        "No hay información de ubicación suficiente para generar el mapa."
    )

# MACHINE LEARNING

st.header("🤖 Machine Learning")

st.markdown(
    """
    Se utilizan modelos de regresión para predecir la concentración
    de PM2.5 a partir de variables relacionadas con el tránsito,
    meteorología, tiempo y otros contaminantes disponibles.
    """
)

# CREAR MODELO

ml = ModeloML(
    df,
    variable_objetivo="pm25"
)

# ENTRENAR Y EVALUAR

try:

    tabla_resultados = ml.entrenar_y_evaluar()

    # Variables utilizadas

    st.subheader("Variables predictoras")

    variables = ml.variables_predictoras

    if variables:
        st.write(
            "Las variables utilizadas por los modelos fueron:"
        )

        st.write(
            ", ".join(variables)
        )

    # Tamaño de los conjuntos

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Datos de entrenamiento",
            f"{len(ml.X_train):,}"
        )

    with col2:
        st.metric(
            "Datos de prueba",
            f"{len(ml.X_test):,}"
        )

    # Resultados

    st.subheader("Comparación de modelos")

    tabla_mostrar = tabla_resultados.copy()

    tabla_mostrar = tabla_mostrar.round(4)

    st.dataframe(
        tabla_mostrar,
        use_container_width=True
    )

    # Mejor modelo

    mejor_nombre, mejor_modelo = ml.elegir_mejor_modelo()

    st.success(
        f"🏆 Mejor modelo según RMSE: **{mejor_nombre}**"
    )

    # Métricas del mejor modelo

    st.subheader("Métricas del mejor modelo")

    resultados_mejor = tabla_resultados.loc[mejor_nombre]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "MAE",
            f"{resultados_mejor['MAE']:.4f}"
        )

    with col2:
        st.metric(
            "MSE",
            f"{resultados_mejor['MSE']:.4f}"
        )

    with col3:
        st.metric(
            "RMSE",
            f"{resultados_mejor['RMSE']:.4f}"
        )

    with col4:
        st.metric(
            "R²",
            f"{resultados_mejor['R2']:.4f}"
        )

 # PREDICCIONES

    st.subheader("Predicciones")

    predicciones = ml.predecir(mejor_nombre)

    resultados_predicciones = pd.DataFrame(
        {
            "Valor real PM2.5": ml.y_test.values,
            "Predicción PM2.5": predicciones,
        }
    )

    st.dataframe(
        resultados_predicciones.head(20).round(3),
        use_container_width=True
    )

except Exception as e:

    st.error(
        f"Ocurrió un error durante el entrenamiento del modelo: {e}"
    )

# INFORMACIÓN FINAL

st.divider()

st.caption(
    "Proyecto Big Data — Persona C: Visualización + Machine Learning"
)