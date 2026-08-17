"""
src/basedatos/GestorBaseDatos.py

Capa de ALMACENAMIENTO (Persona B).

Guarda en SQLite los datos ya LIMPIOS que produce Persona A (Roberto):
clima, calidad del aire y congestión (tráfico real). Luego los une con un
JOIN en SQL, devolviendo el dataset unificado — equivalente al df_final.

Hecho con SQLAlchemy + pandas (como el ejercicio de clase).

    GestorDatos (Roberto)  ->  GestorBaseDatos  ->  ProcesadorEDA  ->  Modelo
      (limpia CSVs)             (guarda + JOIN)      (explora/limpia)    (usa)
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

# Nombres de las tablas en la base.
TABLA_CLIMA = "clima"
TABLA_AIRE = "calidad_aire"
TABLA_CONGESTION = "congestion"


class GestorBaseDatos:
    """Guarda en SQLite las 3 fuentes limpias y las une con un JOIN."""

    def __init__(self, db_path: str = "datos_proyecto.db"):
        self.db_path = str(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    # --- CARGA (guardar cada fuente limpia como una tabla) -----------------

    def guardar_clima(self, df: pd.DataFrame) -> None:
        self._guardar(df, TABLA_CLIMA)

    def guardar_aire(self, df: pd.DataFrame) -> None:
        self._guardar(df, TABLA_AIRE)

    def guardar_congestion(self, df: pd.DataFrame) -> None:
        self._guardar(df, TABLA_CONGESTION)

    def _guardar(self, df: pd.DataFrame, tabla: str) -> None:
        if df is None or df.empty:
            print(f"[GestorBaseDatos] AVISO: DataFrame vacio, no se guardo '{tabla}'.")
            return
        # if_exists="replace" -> cada corrida deja la tabla limpia (sin repetir).
        df.to_sql(tabla, self.engine, if_exists="replace", index=False)
        print(f"[GestorBaseDatos] Guardadas {len(df)} filas en '{tabla}'.")

    # --- CONSULTAS ---------------------------------------------------------

    def obtener_datos_unificados(self) -> pd.DataFrame:
        """
        Une clima + calidad de aire + congestión con un JOIN (misma lógica
        que unir_datos de Roberto) y devuelve el dataset completo.

        - clima + aire: por fecha, hora, día, ubicación y coordenadas.
        - + congestión: por hora, día y ubicación (patrón típico de tráfico).
        """
        query = """
            SELECT
                c.fecha, c.hora, c.dia_semana, c.ubicacion,
                c.temperatura, c.humedad, c.velocidad_viento,
                c.latitud, c.longitud,
                a.pm_10, a.pm_2_5, a.no2, a.o3,
                g.duracion_normal_seg, g.duracion_trafico_seg,
                g.congestion_ratio, g.congestion_ajustada, g.nivel_congestion
            FROM clima c
            LEFT JOIN calidad_aire a
                ON  c.fecha      = a.fecha
                AND c.hora       = a.hora
                AND c.dia_semana = a.dia_semana
                AND c.latitud    = a.latitud
                AND c.longitud   = a.longitud
                AND c.ubicacion  = a.ubicacion
            LEFT JOIN congestion g
                ON  c.hora       = g.hora
                AND c.dia_semana = g.dia_semana
                AND c.ubicacion  = g.ubicacion
        """
        return pd.read_sql(query, self.engine)

    def promedio_por_hora(self, columna: str = "pm_2_5") -> pd.DataFrame:
        """Consulta de ejemplo: promedio de una columna del aire por hora del día."""
        return pd.read_sql(
            f"SELECT hora, AVG({columna}) AS promedio "
            f"FROM {TABLA_AIRE} GROUP BY hora ORDER BY hora",
            self.engine,
        )
