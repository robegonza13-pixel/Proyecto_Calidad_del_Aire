"""
main.py

Punto de entrada principal del proyecto.

Este archivo coordina:
1. Carga de datos.
2. Visualización.
3. Entrenamiento de modelos de Machine Learning.
4. Evaluación y comparación de modelos.
5. Selección del mejor modelo.
6. Guardado del mejor modelo mediante Joblib.
"""

from pathlib import Path
import sys


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORTACIONES DEL PROYECTO
# ============================================================

from helpers.datos_demo import generar_datos_demo
from visualizacion.visualizador import Visualizador
from modelos.modelo_ml import ModeloML


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("PROYECTO BIG DATA - ANÁLISIS DE CONTAMINACIÓN Y TRÁNSITO")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. CARGAR DATOS
    # --------------------------------------------------------

    print("\n[1] Cargando datos...")

    # TEMPORAL:
    # Se utilizan datos DEMO mientras se integra el DataFrame real
    # proporcionado por Persona A/B.

    df = generar_datos_demo()

    print("Datos cargados correctamente.")
    print(f"Cantidad de filas: {len(df)}")
    print(f"Cantidad de columnas: {len(df.columns)}")

    print("\nColumnas disponibles:")
    print(list(df.columns))


    # --------------------------------------------------------
    # 2. VISUALIZACIÓN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[2] GENERANDO VISUALIZACIONES")
    print("=" * 70)

    viz = Visualizador(df)

    # 2.1 Flujo vehicular vs PM2.5
    print("\nGenerando gráfico: Flujo vehicular vs PM2.5...")
    fig_flujo_pm25 = viz.graficar_flujo_vs_pm25()

    if fig_flujo_pm25 is not None:
        fig_flujo_pm25.show()
        print("✓ Gráfico generado correctamente.")
    else:
        print("⚠ No se pudo generar el gráfico.")


    # 2.2 Flujo vehicular vs NO2
    print("\nGenerando gráfico: Flujo vehicular vs NO2...")
    fig_flujo_no2 = viz.graficar_flujo_vs_no2()

    if fig_flujo_no2 is not None:
        fig_flujo_no2.show()
        print("✓ Gráfico generado correctamente.")
    else:
        print("⚠ No se pudo generar el gráfico.")


    # 2.3 Series temporales
    print("\nGenerando series temporales...")
    fig_series = viz.graficar_series_temporales()

    if fig_series is not None:
        fig_series.show()
        print("✓ Serie temporal generada correctamente.")
    else:
        print("⚠ No se pudo generar la serie temporal.")


    # 2.4 PM2.5 vs meteorología
    print("\nGenerando gráfico: PM2.5 vs meteorología...")
    fig_meteorologia = viz.graficar_pm25_vs_meteorologia()

    if fig_meteorologia is not None:
        fig_meteorologia.show()
        print("✓ Gráfico meteorológico generado correctamente.")
    else:
        print("⚠ No se pudo generar el gráfico meteorológico.")


    # 2.5 Heatmap
    print("\nGenerando heatmap de correlaciones...")
    fig_heatmap = viz.graficar_heatmap_correlaciones()

    if fig_heatmap is not None:
        fig_heatmap.show()
        print("✓ Heatmap generado correctamente.")
    else:
        print("⚠ No se pudo generar el heatmap.")


    # 2.6 Mapa
    print("\nGenerando mapa de zonas críticas...")
    fig_mapa = viz.mapa_zonas_criticas()

    if fig_mapa is not None:
        fig_mapa.show()
        print("✓ Mapa generado correctamente.")
    else:
        print("⚠ No se pudo generar el mapa.")


    # --------------------------------------------------------
    # 3. MACHINE LEARNING
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[3] MACHINE LEARNING")
    print("=" * 70)

    print("\nCreando modelo de Machine Learning...")

    ml = ModeloML(
        df,
        variable_objetivo="pm25"
    )

    print("ModeloML creado correctamente.")


    # --------------------------------------------------------
    # 4. ENTRENAR Y EVALUAR MODELOS
    # --------------------------------------------------------

    print("\nEntrenando y evaluando modelos...")

    resultados = ml.entrenar_y_evaluar()

    print("\n✓ Entrenamiento terminado.")

    print("\nVariables predictoras utilizadas:")
    print(ml.variables_predictoras)

    print("\nCantidad de datos:")
    print(f"Entrenamiento: {len(ml.X_train)}")
    print(f"Prueba:        {len(ml.X_test)}")


    # --------------------------------------------------------
    # 5. RESULTADOS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTADOS DE LOS MODELOS")
    print("=" * 70)

    print("\nComparación de modelos:")

    print(
        resultados.round(3)
    )


    # --------------------------------------------------------
    # 6. MEJOR MODELO
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"Mejor modelo según RMSE: {ml.mejor_modelo_nombre}"
    )

    mejor_modelo = ml.modelos_entrenados[
        ml.mejor_modelo_nombre
    ]

    print(
        f"Modelo seleccionado: {mejor_modelo.__class__.__name__}"
    )


    # --------------------------------------------------------
    # 7. GUARDAR MODELO CON JOBLIB
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[4] GUARDANDO MODELO")
    print("=" * 70)

    carpeta_modelos = BASE_DIR / "modelos_guardados"

    carpeta_modelos.mkdir(
        parents=True,
        exist_ok=True
    )

    ruta_modelo = carpeta_modelos / "modelo.pkl"

    ml.guardar_modelo(
        str(ruta_modelo)
    )

    print(
        f"\n✓ Modelo guardado correctamente en:\n"
        f"{ruta_modelo}"
    )


    # --------------------------------------------------------
    # 8. FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)

    print("\nEl proyecto ejecutó correctamente:")
    print("✓ Carga de datos")
    print("✓ Visualizaciones")
    print("✓ Entrenamiento de modelos")
    print("✓ Evaluación de modelos")
    print("✓ Comparación de modelos")
    print("✓ Selección del mejor modelo")
    print("✓ Guardado del modelo mediante Joblib")
    print("\nArchivo generado:")
    print(f"  {ruta_modelo}")


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()