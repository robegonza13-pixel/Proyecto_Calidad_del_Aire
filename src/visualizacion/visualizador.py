"""
src/visualizacion/visualizador.py

Todas las funciones devuelven un objeto plotly.graph_objects.Figure (o
None si a la operación le faltan columnas todavía), para poder usarse
tanto en un script normal (fig.show()) como dentro del dashboard de
Streamlit (st.plotly_chart(fig)).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helpers.columnas import CONTAMINANTES, METEOROLOGICAS, obtener_columna, validar_columnas_requeridas


class Visualizador:
    """
    Genera las visualizaciones del proyecto a partir de un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Datos ya limpios (reales o de ejemplo).
    column_mapping : dict, opcional
        Permite remapear nombres semánticos ("pm25", "flujo_vehicular", ...)
        a los nombres reales de columna del DataFrame final, sin tener que
        modificar esta clase. Ver helpers/columnas.py.
    """

    def __init__(self, df: pd.DataFrame, column_mapping: dict | None = None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Visualizador espera un pandas DataFrame.")
        self.df = df.copy()
        self.mapping = column_mapping or {}


    # Utilidades internas

    def _col(self, nombre_semantico: str) -> str:
        return obtener_columna(nombre_semantico, self.mapping)

    def _listos(self, requeridas: list[str], contexto: str) -> bool:
        return validar_columnas_requeridas(self.df, requeridas, self.mapping, contexto)


    # 1. Flujo vehicular vs PM2.5
    # Columnas requeridas: 'flujo_vehicular', 'pm25'

    def graficar_flujo_vs_pm25(self) -> go.Figure | None:
        if not self._listos(["flujo_vehicular", "pm25"], "flujo_vs_pm25"):
            return None
        col_x, col_y = self._col("flujo_vehicular"), self._col("pm25")
        r = self.df[[col_x, col_y]].corr().iloc[0, 1]
        return px.scatter(
            self.df,
            x=col_x,
            y=col_y,
            opacity=0.6,
            labels={col_x: "Flujo vehicular", col_y: "PM2.5 (µg/m³)"},
            title=f"Relación entre flujo vehicular y PM2.5 (r = {r:.2f})",
        )


    # 2. Flujo vehicular vs NO2
    # Columnas requeridas: 'flujo_vehicular', 'no2'

    def graficar_flujo_vs_no2(self) -> go.Figure | None:
        if not self._listos(["flujo_vehicular", "no2"], "flujo_vs_no2"):
            return None
        col_x, col_y = self._col("flujo_vehicular"), self._col("no2")
        r = self.df[[col_x, col_y]].corr().iloc[0, 1]
        return px.scatter(
            self.df,
            x=col_x,
            y=col_y,
            opacity=0.6,
            labels={col_x: "Flujo vehicular", col_y: "NO2 (µg/m³)"},
            title=f"Relación entre flujo vehicular y NO2 (r = {r:.2f})",
        )


    # 3. Series temporales de contaminación
    # Columnas requeridas: 'fecha_hora' + al menos un contaminante

    def graficar_series_temporales(self, contaminantes: list[str] | None = None) -> go.Figure | None:
        contaminantes = contaminantes or CONTAMINANTES
        if not self._listos(["fecha_hora"], "series_temporales"):
            return None

        disponibles = [c for c in contaminantes if self._listos([c], "series_temporales")]
        if not disponibles:
            print("[series_temporales] Ningún contaminante solicitado está disponible todavía.")
            return None

        col_fecha = self._col("fecha_hora")
        cols_cont = [self._col(c) for c in disponibles]
        df_plot = self.df[[col_fecha] + cols_cont].copy()
        df_plot[col_fecha] = pd.to_datetime(df_plot[col_fecha])
        df_plot = df_plot.sort_values(col_fecha)

        fig = go.Figure()
        for c, col_real in zip(disponibles, cols_cont):
            fig.add_trace(go.Scatter(x=df_plot[col_fecha], y=df_plot[col_real], mode="lines", name=c.upper()))
        fig.update_layout(
            title="Serie temporal de contaminación",
            xaxis_title="Fecha y hora",
            yaxis_title="Concentración",
        )
        return fig


    # 4. PM2.5 frente a variables meteorológicas
    # Columnas requeridas: 'pm25' + al menos una de temperatura/humedad/viento

    def graficar_pm25_vs_meteorologia(self) -> go.Figure | None:
        if not self._listos(["pm25"], "pm25_vs_meteorologia"):
            return None
        disponibles = [v for v in METEOROLOGICAS if self._listos([v], "pm25_vs_meteorologia")]
        if not disponibles:
            print("[pm25_vs_meteorologia] No hay variables meteorológicas disponibles todavía.")
            return None

        col_pm25 = self._col("pm25")
        etiquetas = {"temperatura": "Temperatura (°C)", "humedad": "Humedad (%)", "viento": "Viento (m/s)"}

        titulos = []
        for v in disponibles:
            r = self.df[[self._col(v), col_pm25]].corr().iloc[0, 1]
            titulos.append(f"{etiquetas[v]} (r = {r:.2f})")

        fig = make_subplots(rows=1, cols=len(disponibles), subplot_titles=titulos)
        for i, var in enumerate(disponibles, start=1):
            col_var = self._col(var)
            fig.add_trace(
                go.Scatter(x=self.df[col_var], y=self.df[col_pm25], mode="markers", opacity=0.6, showlegend=False),
                row=1,
                col=i,
            )
            fig.update_xaxes(title_text=etiquetas[var], row=1, col=i)
            if i == 1:
                fig.update_yaxes(title_text="PM2.5 (µg/m³)", row=1, col=i)

        fig.update_layout(title="PM2.5 frente a variables meteorológicas")
        return fig


    # 5. Heatmap de correlaciones
    # Columnas requeridas: al menos 2 columnas numéricas del contrato

    def graficar_heatmap_correlaciones(self, columnas: list[str] | None = None) -> go.Figure | None:
        nombres_semanticos = columnas or (CONTAMINANTES + METEOROLOGICAS + ["flujo_vehicular"])
        disponibles = [n for n in nombres_semanticos if self._listos([n], "heatmap")]

        if len(disponibles) < 2:
            print("[heatmap] Se necesitan al menos 2 columnas numéricas disponibles para el heatmap.")
            return None

        cols_reales = [self._col(n) for n in disponibles]
        corr = self.df[cols_reales].corr(numeric_only=True)
        corr.index = disponibles
        corr.columns = disponibles

        return px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Heatmap de correlaciones",
        )


    # 6. Visualización preparatoria para el mapa de zonas críticas
    # Requiere 'zona' y/o 'latitud'+'longitud'

    def mapa_zonas_criticas(self, metrica: str = "pm25", agregacion: str = "mean") -> go.Figure | None:
        """
        Si el DataFrame trae 'zona', agrupa por zona (y usa el promedio de
        latitud/longitud de cada zona si existen, para ubicarla en el
        mapa). Si solo hay coordenadas sin 'zona', grafica cada
        observación individual. Si no hay ninguna columna de ubicación,
        avisa y devuelve None en vez de fallar.
        """
        if not self._listos([metrica], "mapa_zonas_criticas"):
            return None
        col_metrica = self._col(metrica)

        tiene_zona = self._listos(["zona"], "mapa_zonas_criticas")
        tiene_coordenadas = self._listos(["latitud", "longitud"], "mapa_zonas_criticas")

        if not tiene_zona and not tiene_coordenadas:
            print("[mapa_zonas_criticas] Aún no hay columnas de ubicación ('zona' o 'latitud'/'longitud').")
            return None

        col_zona = self._col("zona") if tiene_zona else None

        if tiene_zona:
            agg_dict = {col_metrica: agregacion}
            if tiene_coordenadas:
                col_lat, col_lon = self._col("latitud"), self._col("longitud")
                agg_dict[col_lat] = "mean"
                agg_dict[col_lon] = "mean"
            resumen = self.df.groupby(col_zona, as_index=False).agg(agg_dict)
        else:
            col_lat, col_lon = self._col("latitud"), self._col("longitud")
            resumen = self.df[[col_lat, col_lon, col_metrica]].copy()

        if tiene_coordenadas:
            col_lat, col_lon = self._col("latitud"), self._col("longitud")
            fig = px.scatter_mapbox(
                resumen,
                lat=col_lat,
                lon=col_lon,
                color=col_metrica,
                size=col_metrica,
                hover_name=col_zona,
                color_continuous_scale="YlOrRd",
                zoom=9,
                mapbox_style="open-street-map",
                title=f"Zonas críticas — {metrica.upper()} ({agregacion})",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            return fig

        # Sin coordenadas: alternativa temporal en barras por zona.
        resumen = resumen.sort_values(col_metrica, ascending=False)
        fig = px.bar(
            resumen,
            x=col_zona,
            y=col_metrica,
            title=f"{metrica.upper()} por zona ({agregacion}) — alternativa sin coordenadas",
            labels={col_zona: "Zona", col_metrica: metrica.upper()},
        )
        print(
            "[mapa_zonas_criticas] No hay coordenadas todavía; se muestra un ranking por "
            "zona. Cuando existan 'latitud'/'longitud', esta misma función generará el mapa real."
        )
        return fig


    # 7. PM2.5 promedio por hora del día
    # Columnas requeridas: 'pm25', 'hora'

    def graficar_pm25_por_hora(self) -> go.Figure | None:
        """Cómo cambia el PM2.5 a lo largo del día (patrón horario)."""
        if not self._listos(["pm25", "hora"], "pm25_por_hora"):
            return None
        col_pm25, col_hora = self._col("pm25"), self._col("hora")
        resumen = self.df.groupby(col_hora, as_index=False)[col_pm25].mean()
        return px.line(
            resumen,
            x=col_hora,
            y=col_pm25,
            markers=True,
            labels={col_hora: "Hora del día", col_pm25: "PM2.5 (µg/m³)"},
            title="PM2.5 promedio por hora del día",
        )


    # 8. PM2.5 promedio según el nivel de congestión del tráfico
    # Requiere 'pm25' y la columna 'nivel_congestion' (presente en los datos reales)

    def graficar_pm25_por_congestion(self, col_congestion: str = "nivel_congestion") -> go.Figure | None:
        """Compara el PM2.5 promedio en cada nivel de tráfico (Fluido→Pesado)."""
        if not self._listos(["pm25"], "pm25_por_congestion") or col_congestion not in self.df.columns:
            return None
        col_pm25 = self._col("pm25")
        orden = ["Fluido", "Leve", "Moderado", "Pesado"]
        resumen = self.df.groupby(col_congestion, as_index=False)[col_pm25].mean()
        return px.bar(
            resumen,
            x=col_congestion,
            y=col_pm25,
            color=col_pm25,
            color_continuous_scale="YlOrRd",
            category_orders={col_congestion: orden},
            labels={col_congestion: "Nivel de congestión", col_pm25: "PM2.5 promedio (µg/m³)"},
            title="PM2.5 promedio según el nivel de congestión del tráfico",
        )


if __name__ == "__main__":
    # Prueba rápida con datos DEMO (ver helpers/datos_demo.py).
    # Esto NO se ejecuta cuando la clase se importa desde otro módulo.
    from helpers.datos_demo import generar_datos_demo

    print("Probando Visualizador con datos DEMO (no reales)...\n")
    df_demo = generar_datos_demo()
    viz = Visualizador(df_demo)

    pruebas = [
        ("flujo_vs_pm25", viz.graficar_flujo_vs_pm25),
        ("flujo_vs_no2", viz.graficar_flujo_vs_no2),
        ("series_temporales", viz.graficar_series_temporales),
        ("pm25_vs_meteorologia", viz.graficar_pm25_vs_meteorologia),
        ("heatmap_correlaciones", viz.graficar_heatmap_correlaciones),
        ("mapa_zonas_criticas", viz.mapa_zonas_criticas),
    ]
    for nombre, funcion in pruebas:
        resultado = funcion()
        estado = "OK" if resultado is not None else "OMITIDO"
        print(f"  {nombre}: {estado}")

    print("\nListo. Todas las figuras se generaron sin errores sobre datos de ejemplo.")
