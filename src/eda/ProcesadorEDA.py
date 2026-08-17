"""
src/eda/ProcesadorEDA.py

Análisis exploratorio (EDA) y LIMPIEZA fina (Persona B).

Toma el DataFrame unificado real (clima + aire + congestión que arma
GestorBaseDatos) y entrega: estadística descriptiva, correlaciones y una
limpieza fina, dejándolo listo para el modelo.

No inventa datos: el tráfico es la CONGESTIÓN real (congestion_ajustada),
y el CO no existe en los datos reales, así que simplemente no se usa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Puente entre los nombres reales (Roberto) y los nombres "semánticos" que
# usan ModeloML/Visualizador (helpers/columnas.py). Se pasa como
# column_mapping, así NO hay que renombrar columnas ni tocar el modelo.
# 'co' no tiene equivalente real -> el modelo lo omite (nada sintético).
MAPEO_MODELO = {
    "pm25": "pm_2_5",
    "pm10": "pm_10",
    "no2": "no2",
    "o3": "o3",
    "temperatura": "temperatura",
    "humedad": "humedad",
    "viento": "velocidad_viento",
    "flujo_vehicular": "congestion_ajustada",  # el tráfico REAL es la congestión
    "fecha_hora": "fecha",
    "hora": "hora",
    "dia_semana": "dia_semana",
    "latitud": "latitud",
    "longitud": "longitud",
    "zona": "ubicacion",
}


class ProcesadorEDA:
    """Explora y limpia el DataFrame unificado real."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # --- EDA (exploración) -------------------------------------------------

    def resumen(self) -> dict:
        """Nulos, duplicados, estadística descriptiva y correlaciones."""
        numericas = self.df.select_dtypes(include=[np.number])
        return {
            "filas": len(self.df),
            "info_nulos": self.df.isnull().sum().to_dict(),
            "duplicados": int(self.df.duplicated().sum()),
            "estadisticas": self.df.describe().to_dict(),
            "correlaciones": numericas.corr().to_dict(),
        }

    def correlacion_con(self, objetivo: str = "pm_2_5") -> dict:
        """
        Correlación de cada variable numérica con el objetivo.
        Sirve para ver la relación tráfico (congestión) vs PM2.5 / NO2 que
        pide el proyecto.
        """
        numericas = self.df.select_dtypes(include=[np.number])
        if objetivo not in numericas.columns:
            return {}
        return numericas.corr()[objetivo].sort_values(ascending=False).to_dict()

    # --- LIMPIEZA fina -----------------------------------------------------

    def limpiar(self) -> pd.DataFrame:
        """
        Limpieza fina sobre datos ya cerca de limpios (Roberto hizo la base):
        - quita duplicados exactos,
        - convierte 'fecha' a datetime,
        - imputa nulos numéricos con la mediana (por si el JOIN dejó huecos).
        Devuelve el DataFrame con los nombres REALES (sin renombrar).
        """
        if self.df.empty:
            return self.df.copy()

        df = self.df.drop_duplicates().copy()

        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

        numericas = df.select_dtypes(include=[np.number]).columns
        df[numericas] = df[numericas].fillna(df[numericas].median())

        return df
