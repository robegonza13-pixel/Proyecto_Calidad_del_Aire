"""
src/main.py

Orquestador del proyecto — conecta todas las partes con DATOS REALES:

    GestorDatos (Roberto)   carga y limpia los CSV (clima, aire, congestión)
        v
    GestorBaseDatos (Persona B)   guarda en SQLite y une con un JOIN
        v
    ProcesadorEDA (Persona B)     estadística, correlaciones, limpieza fina
        v
    ModeloML (Persona C)          entrena y evalúa la predicción de PM2.5

Ejecutar desde la raíz del proyecto:
    python src/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# La raíz del proyecto y src en el path (los paquetes usan 'src.' y 'helpers.').
RAIZ = Path(__file__).resolve().parent.parent
for _p in (str(RAIZ), str(RAIZ / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.basedatos.GestorBaseDatos import GestorBaseDatos
from src.datos.GestorDatos import GestorDatos
from src.eda.ProcesadorEDA import MAPEO_MODELO, ProcesadorEDA
from src.modelos.ModeloML import ModeloML

DB_PATH = str(RAIZ / "data" / "datos_proyecto.db")

# CSV limpios que dejó Roberto en data/raw
CSV_CLIMA = "df_clima_limpio_san_jose.csv"
CSV_AIRE = "df_aire_limpio_san_jose.csv"
CSV_CONGESTION = "df_congestion_limpio_san_jose.csv"


def main() -> None:
    print("=" * 70)
    print("PIPELINE (datos REALES): CSV -> SQLite -> EDA -> Modelo")
    print("=" * 70)

    # 1. DATOS: cargar los CSV limpios reales (Persona A / Roberto)
    print("\n[1/4] Cargando CSV limpios (Roberto)...")
    gestor_datos = GestorDatos(
        ruta_raw=str(RAIZ / "data" / "raw"),
        ruta_processed=str(RAIZ / "data" / "processed"),
    )
    df_clima = gestor_datos.carga_datos(CSV_CLIMA)
    df_aire = gestor_datos.carga_datos(CSV_AIRE)
    df_congestion = gestor_datos.carga_datos(CSV_CONGESTION)
    print(f"      Clima {len(df_clima)} | Aire {len(df_aire)} | Congestion {len(df_congestion)}")

    # 2. ALMACENAMIENTO: guardar en SQLite y unir (Persona B)
    print("\n[2/4] Guardando en SQLite y unificando (JOIN)...")
    gestor_db = GestorBaseDatos(DB_PATH)
    gestor_db.guardar_clima(df_clima)
    gestor_db.guardar_aire(df_aire)
    gestor_db.guardar_congestion(df_congestion)
    df_unificado = gestor_db.obtener_datos_unificados()
    print(f"      Unificado: {df_unificado.shape[0]} filas, {df_unificado.shape[1]} columnas")

    # 3. EDA + LIMPIEZA fina (Persona B)
    print("\n[3/4] Explorando y limpiando (EDA)...")
    eda = ProcesadorEDA(df_unificado)
    resumen = eda.resumen()
    print(f"      Nulos totales: {sum(resumen['info_nulos'].values())} | Duplicados: {resumen['duplicados']}")
    print("      Correlacion con PM2.5 (top):")
    for nombre, valor in list(eda.correlacion_con("pm_2_5").items())[:6]:
        print(f"        {nombre}: {round(valor, 3)}")
    df_limpio = eda.limpiar()

    # 4. MODELO: entrenar con datos REALES (column_mapping, sin nada sintético)
    print("\n[4/4] Entrenando modelos de regresion (PM2.5)...")
    ml = ModeloML(df_limpio, variable_objetivo="pm25", column_mapping=MAPEO_MODELO)
    tabla = ml.entrenar_y_evaluar()
    print(f"      Variables usadas: {ml.variables_predictoras}")
    print(tabla.round(3))
    print(f"      Mejor modelo (RMSE): {ml.mejor_modelo_nombre}")

    print("\nListo: TODO corre con datos REALES. Solo falta visualizacion.py (Persona C).")


if __name__ == "__main__":
    main()
