"""
src/eda/explorar.py

Exploración VISUAL del EDA (Persona B).

Toma los datos reales guardados en SQLite, los limpia con ProcesadorEDA y
genera los gráficos del análisis exploratorio con matplotlib/seaborn
(correlaciones, tráfico vs contaminantes, patrón temporal, distribución).
Los PNG se guardan en data/processed/.

Ejecutar desde la raíz del proyecto:
    python src/eda/explorar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: guarda a archivo
import matplotlib.pyplot as plt
import seaborn as sns

RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (str(RAIZ), str(RAIZ / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.basedatos.GestorBaseDatos import GestorBaseDatos
from src.eda.ProcesadorEDA import ProcesadorEDA

CARPETA_GRAFICOS = RAIZ / "data" / "processed"
DB_PATH = str(RAIZ / "data" / "datos_proyecto.db")


def cargar_datos_limpios():
    """Lee el JOIN de SQLite y devuelve el DataFrame limpio (datos reales)."""
    gestor = GestorBaseDatos(DB_PATH)
    df = gestor.obtener_datos_unificados()
    if df.empty:
        return None
    return ProcesadorEDA(df).limpiar()


def graficar(df) -> None:
    CARPETA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Mapa de calor de correlaciones
    plt.figure(figsize=(11, 9))
    sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Mapa de correlaciones")
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS / "1_correlaciones.png", dpi=120)
    plt.close()

    # 2. Dispersión: congestión (tráfico real) vs PM2.5 y vs NO2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(data=df, x="congestion_ajustada", y="pm_2_5", ax=axes[0], alpha=0.4)
    axes[0].set_title("Congestion vs PM2.5")
    sns.scatterplot(data=df, x="congestion_ajustada", y="no2", ax=axes[1], alpha=0.4, color="darkorange")
    axes[1].set_title("Congestion vs NO2")
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS / "2_trafico_vs_contaminantes.png", dpi=120)
    plt.close()

    # 3. Patrón temporal: PM2.5 promedio por hora y por día de semana
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df.groupby("hora")["pm_2_5"].mean().plot(marker="o", ax=axes[0])
    axes[0].set_title("PM2.5 promedio por hora del dia")
    axes[0].set_xlabel("Hora"); axes[0].set_ylabel("PM2.5")
    dias = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
    df.groupby("dia_semana")["pm_2_5"].mean().plot(kind="bar", ax=axes[1], color="seagreen")
    axes[1].set_title("PM2.5 promedio por dia de la semana")
    axes[1].set_xticklabels(dias, rotation=0)
    axes[1].set_xlabel("Dia"); axes[1].set_ylabel("PM2.5")
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS / "3_patron_temporal.png", dpi=120)
    plt.close()

    # 4. Distribución de PM2.5
    plt.figure(figsize=(9, 5))
    sns.histplot(df["pm_2_5"], kde=True, color="steelblue")
    plt.title("Distribucion de PM2.5")
    plt.xlabel("PM2.5 (ug/m3)")
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS / "4_distribucion_pm25.png", dpi=120)
    plt.close()

    print(f"Graficos guardados en: {CARPETA_GRAFICOS}")


def main() -> None:
    df = cargar_datos_limpios()
    if df is None:
        print("No hay datos en la base. Corre primero: python src/main.py")
        return

    print("=== info ===")
    df.info()
    print("\n=== describe ===")
    print(df.describe().round(2))
    print("\n=== correlacion con PM2.5 ===")
    print(df.select_dtypes("number").corr()["pm_2_5"].round(3).sort_values(ascending=False))

    graficar(df)


if __name__ == "__main__":
    main()
