"""
helpers/datos_demo.py

Generador de datos SINTÉTICOS usados ÚNICAMENTE para verificar que
Visualizador y ModeloML funcionan.

Este módulo se deja de usar (o se aísla solo para pruebas) en cuanto el
DataFrame real de Persona A/B esté disponible; ni Visualizador ni
ModeloML lo importan — solo lo usan sus bloques de prueba (__main__) y
el dashboard, como último recurso si no encuentra datos reales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generar_datos_demo(n_filas: int = 24 * 30, semilla: int = 42) -> pd.DataFrame:
    """
    Crea un DataFrame de ejemplo con la misma forma (columnas) que se
    espera del DataFrame final, para probar Visualizador y ModeloML de
    forma aislada.

    Parameters
    ----------
    n_filas : int
        Cantidad de registros a generar (por defecto, 30 días en
        resolución horaria).
    semilla : int
        Semilla del generador aleatorio (reproducibilidad).

    Returns
    -------
    pd.DataFrame
        Datos SINTÉTICOS, no reales.
    """
    rng = np.random.default_rng(semilla)

    fechas = pd.date_range("2025-01-01", periods=n_filas, freq="h")
    hora = fechas.hour
    dia_semana = fechas.dayofweek  # 0 = lunes ... 6 = domingo

    # Flujo vehicular: más alto en horas pico (7-9h y 16-19h) entre semana.
    es_hora_pico = np.isin(hora, [7, 8, 9, 16, 17, 18, 19])
    es_entre_semana = dia_semana < 5
    base_flujo = 200 + 600 * es_hora_pico * es_entre_semana
    flujo_vehicular = np.clip(base_flujo + rng.normal(0, 60, n_filas), 20, None)

    # Meteorología sintética con un ciclo diario simple.
    temperatura = 20 + 5 * np.sin((hora - 8) / 24 * 2 * np.pi) + rng.normal(0, 1.5, n_filas)
    humedad = np.clip(70 - 15 * np.sin((hora - 8) / 24 * 2 * np.pi) + rng.normal(0, 5, n_filas), 20, 100)
    viento = np.clip(rng.normal(8, 3, n_filas), 0, None)

    # PM2.5: relación artificial positiva con tráfico y humedad, negativa
    # con viento (a más viento, se dispersa más el contaminante).
    ruido = rng.normal(0, 4, n_filas)
    pm25 = np.clip(10 + 0.02 * flujo_vehicular + 0.15 * humedad - 0.8 * viento + ruido, 2, None)

    pm10 = np.clip(pm25 * 1.6 + rng.normal(0, 5, n_filas), 3, None)
    no2 = np.clip(0.03 * flujo_vehicular + rng.normal(15, 5, n_filas), 1, None)
    co = np.clip(0.0006 * flujo_vehicular + rng.normal(0.4, 0.15, n_filas), 0.05, None)
    o3 = np.clip(40 - 0.1 * no2 + rng.normal(0, 6, n_filas), 1, None)

    # Ubicación: puntos alrededor de 4 zonas de referencia del GAM
    # (coordenadas aproximadas, solo para ejercitar el código del mapa).
    zonas_ref = {
        "San José Centro": (9.9333, -84.0833),
        "Cartago Centro": (9.8644, -83.9194),
        "Heredia Centro": (9.9989, -84.1169),
        "Alajuela Centro": (10.0167, -84.2167),
    }
    nombres_zona = rng.choice(list(zonas_ref.keys()), size=n_filas)
    latitud = np.array([zonas_ref[z][0] for z in nombres_zona]) + rng.normal(0, 0.01, n_filas)
    longitud = np.array([zonas_ref[z][1] for z in nombres_zona]) + rng.normal(0, 0.01, n_filas)

    return pd.DataFrame(
        {
            "fecha_hora": fechas,
            "hora": hora,
            "dia_semana": dia_semana,
            "flujo_vehicular": flujo_vehicular.round(1),
            "temperatura": temperatura.round(1),
            "humedad": humedad.round(1),
            "velocidad_viento": viento.round(1),
            "pm25": pm25.round(2),
            "pm10": pm10.round(2),
            "no2": no2.round(2),
            "co": co.round(3),
            "o3": o3.round(2),
            "zona": nombres_zona,
            "latitud": latitud.round(5),
            "longitud": longitud.round(5),
        }
    )


if __name__ == "__main__":
    df_demo = generar_datos_demo()
    print("Datos DEMO generados (NO reales):")
    print(df_demo.head())
    print(f"\nForma: {df_demo.shape}")