"""
helpers/columnas.py

Este módulo define, en un solo lugar, los nombres de columna "semánticos"
que usan Visualizador y ModeloML, junto con el nombre de columna que se
espera por defecto en el DataFrame final.El módulo no asume directamente
que una columna existe con un nombre fijo: en su lugar, consulta este
contrato (que se puede sobreescribir con un `column_mapping`).

Si el DataFrame final usa nombres distintos, NO hay que tocar Visualizador
ni ModeloML: basta con pasar un diccionario `column_mapping` al crearlos.

    mapping = {"pm25": "PM2_5", "flujo_vehicular": "vehiculos_hora"}
    viz = Visualizador(df, column_mapping=mapping)
    ml = ModeloML(df, column_mapping=mapping)
"""

from __future__ import annotations

import pandas as pd

# Nombre semántico -> nombre de columna por defecto que se espera en el
# DataFrame final
COLUMNAS_ESPERADAS: dict[str, str] = {
    # Contaminantes (calidad del aire, Open-Meteo u otra fuente similar)
    "pm25": "pm25",
    "pm10": "pm10",
    "no2": "no2",
    "co": "co",
    "o3": "o3",
    # Variables meteorológicas (Open-Meteo histórico)
    "temperatura": "temperatura",
    "humedad": "humedad",
    "viento": "velocidad_viento",
    # Tránsito (MOPT / COSEVI)
    "flujo_vehicular": "flujo_vehicular",
    # Tiempo
    "fecha_hora": "fecha_hora",
    "hora": "hora",
    "dia_semana": "dia_semana",
    # Ubicación (opcional; puede no venir en la primera entrega)
    "latitud": "latitud",
    "longitud": "longitud",
    "zona": "zona",
}

# Contaminantes y variables meteorológicas, para no repetir la lista en
# cada módulo que las necesite.
CONTAMINANTES = ["pm25", "pm10", "no2", "co", "o3"]
METEOROLOGICAS = ["temperatura", "humedad", "viento"]


def obtener_columna(nombre_semantico: str, column_mapping: dict | None = None) -> str:
    """
    Traduce un nombre semántico (p. ej. "pm25") al nombre real de columna
    que debe buscarse en el DataFrame, respetando cualquier sobreescritura
    indicada en `column_mapping`.
    """
    if column_mapping and nombre_semantico in column_mapping:
        return column_mapping[nombre_semantico]
    if nombre_semantico not in COLUMNAS_ESPERADAS:
        raise KeyError(
            f"'{nombre_semantico}' no es un nombre semántico reconocido. "
            f"Opciones válidas: {list(COLUMNAS_ESPERADAS)}"
        )
    return COLUMNAS_ESPERADAS[nombre_semantico]


def columnas_disponibles(
    df: pd.DataFrame,
    nombres_semanticos: list[str],
    column_mapping: dict | None = None,
) -> dict[str, bool]:
    """Indica, para cada nombre semántico pedido, si existe en `df`."""
    disponibilidad = {}
    for nombre in nombres_semanticos:
        col_real = obtener_columna(nombre, column_mapping)
        disponibilidad[nombre] = col_real in df.columns
    return disponibilidad


def validar_columnas_requeridas(
    df: pd.DataFrame,
    requeridas: list[str],
    column_mapping: dict | None = None,
    contexto: str = "",
) -> bool:
    """
    Revisa si todas las columnas semánticas en `requeridas` están en `df`.
    No lanza excepción: devuelve True/False e imprime una advertencia
    clara cuando faltan columnas, para que Visualizador/ModeloML puedan
    omitir un gráfico o feature en vez de fallar con un traceback.
    """
    disponibilidad = columnas_disponibles(df, requeridas, column_mapping)
    faltantes = [
        obtener_columna(nombre, column_mapping)
        for nombre, existe in disponibilidad.items()
        if not existe
    ]
    if faltantes:
        prefijo = f"[{contexto}] " if contexto else ""
        print(
            f"{prefijo}AVISO: faltan columnas en el DataFrame: {faltantes}. "
            "Esta operacion se omite hasta que esas columnas esten disponibles."
        )
        return False
    return True