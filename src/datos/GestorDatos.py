import os
import pandas as pd


class GestorDatos:
    def __init__(self, ruta_raw="../data/raw",
                 ruta_processed="../data/processed"):

        self.__ruta_raw = ruta_raw
        self.__ruta_processed = ruta_processed

        os.makedirs(self.__ruta_raw, exist_ok=True)
        os.makedirs(self.__ruta_processed, exist_ok=True)

    def carga_datos(self, nombre_archivo):
        df = pd.read_csv(
            self.__ruta_raw + "/" + nombre_archivo
        )
        return df

    def guardar_csv(self, df, nombre_archivo, procesado=False):
        ruta = self.__ruta_processed if procesado else self.__ruta_raw

        df.to_csv(
            ruta + "/" + nombre_archivo,
            index=False,
            encoding="utf-8"
        )
    @staticmethod
    def limpiar_clima(df, ubicacion):
        df = df.copy()

        df.columns = ["fecha", "temperatura", "humedad", "velocidad_viento", "latitud", "longitud"]

        df["fecha"] = pd.to_datetime(df["fecha"])

        df["hora"] = df["fecha"].dt.hour
        df["dia_semana"] = df["fecha"].dt.dayofweek

        df["ubicacion"] = ubicacion

        columnas = [
            "fecha",
            "hora",
            "dia_semana",
            "ubicacion",
            "temperatura",
            "humedad",
            "velocidad_viento",
            "latitud",
            "longitud"
        ]

        return df[columnas]

    @staticmethod
    def limpiar_calidad_aire(df, ubicacion):
        df = df.copy()

        df.columns = ["fecha", "pm_10", "pm_2_5", "no2", "o3", "latitud", "longitud"]

        df["fecha"] = pd.to_datetime(df["fecha"])

        df["hora"] = df["fecha"].dt.hour
        df["dia_semana"] = df["fecha"].dt.dayofweek

        df["ubicacion"] = ubicacion

        columnas = [
            "fecha",
            "hora",
            "dia_semana",
            "ubicacion",
            "pm_10",
            "pm_2_5",
            "no2",
            "o3",
            "latitud",
            "longitud"
        ]

        return df[columnas]

    @staticmethod
    def limpiar_congestion(df, ubicacion):
        df = df.copy()

        df.columns = ["origen_lat", "origen_lon", "destino_lat", "destino_lon", "departure_time", "duracion_normal_seg",
            "duracion_trafico_seg", "congestion_ratio"]

        df["fecha"] = pd.to_datetime(df["departure_time"], utc=True)

        df["hora"] = df["fecha"].dt.hour
        df["dia_semana"] = df["fecha"].dt.dayofweek

        df["ubicacion"] = ubicacion

        df["congestion_ajustada"] = df["congestion_ratio"].clip(lower=1)

        df["nivel_congestion"] = pd.cut(
            df["congestion_ajustada"],
            bins=[0, 1, 1.3, 1.7, float("inf")],
            labels=["Fluido", "Leve", "Moderado", "Pesado"],
            include_lowest=True
        )

        columnas = [
            "hora",
            "dia_semana",
            "ubicacion",
            "origen_lat",
            "origen_lon",
            "destino_lat",
            "destino_lon",
            "duracion_normal_seg",
            "duracion_trafico_seg",
            "congestion_ratio",
            "congestion_ajustada",
            "nivel_congestion"
        ]

        return df[columnas]

    @staticmethod
    def unir_datos(df_clima, df_calidad_aire, df_congestion):
        df = pd.merge(df_clima, df_calidad_aire, on=["fecha", "hora", "dia_semana", "latitud", "longitud", "ubicacion"],
                      how="left")
        df_final = pd.merge(df, df_congestion, on=["hora", "dia_semana", "ubicacion"], how="left")
        return df_final