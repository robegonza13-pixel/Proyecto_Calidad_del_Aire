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

    def limpiar_clima(self, df):
        # limpieza específica del clima
        return df

    def limpiar_calidad_aire(self, df):
        # limpieza específica del aire
        return df

    def limpiar_congestion(self, df):
        # limpieza específica del tráfico
        return df

    def unir_datos(df_clima, df_calidad_aire, df_congestion)
        return df
