# Proyecto 3 — Análisis y Predicción de la Calidad del Aire en el GAM

Analiza la calidad del aire en la Gran Área Metropolitana (Costa Rica) y la
relaciona con el **tráfico vehicular** y las **condiciones meteorológicas**
para **predecir la concentración de PM2.5**.

Pipeline completo, con **datos reales**, de punta a punta:

```
CSV limpios  →  SQLite (JOIN)  →  EDA / limpieza  →  Modelo ML  →  Dashboard
 GestorDatos     GestorBaseDatos    ProcesadorEDA      ModeloML      Visualizador
```

## Estructura del proyecto

```
Proyecto_Calidad_del_Aire/
├── src/
│   ├── datos/           # GestorDatos  — carga y limpia los CSV
│   ├── basedatos/       # GestorBaseDatos — SQLite: guarda y une (JOIN)
│   ├── api/             # ClienteAPI — descarga de Open-Meteo
│   ├── eda/             # ProcesadorEDA — estadística, correlaciones, limpieza
│   │                    #   + explorar.py (gráficos del EDA)
│   ├── visualizacion/   # Visualizador — gráficos plotly
│   │                    #   + app.py (dashboard Streamlit)
│   ├── modelos/         # ModeloML — regresión (lineal, KNN, Random Forest)
│   ├── helpers/         # columnas.py (contrato) y datos_demo.py
│   └── main.py          # Orquestador del pipeline
├── notebooks/
│   └── exploracion_inicial.ipynb
├── data/
│   ├── raw/             # CSV crudos y limpios (clima, aire, congestión)
│   └── processed/       # df_final.csv, modelo.pkl, gráficos del EDA
├── requirements.txt
└── README.md
```

## Requisitos e instalación

Requiere **Python 3.10+**. Desde la raíz del proyecto:

```bash
# 1. (opcional) crear/activar entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell

# 2. instalar dependencias
pip install -r requirements.txt
```

## Cómo ejecutar

**Pipeline completo** (carga datos reales → SQLite → EDA → entrena el modelo):

```bash
python src/main.py
```

**Dashboard interactivo** (visualizaciones + modelo, con datos reales):

```bash
streamlit run src/visualizacion/app.py
```

**Gráficos del EDA** (se guardan en `data/processed/`):

```bash
python src/eda/explorar.py
```

## Módulos (cada uno es una clase)

| Carpeta | Clase | Responsabilidad |
|---|---|---|
| `datos/` | `GestorDatos` | Cargar, limpiar y exportar CSV (clima, aire, congestión) |
| `basedatos/` | `GestorBaseDatos` | Guardar en SQLite y unir las fuentes con un JOIN |
| `api/` | `ClienteAPI` | Peticiones a la API pública de Open-Meteo |
| `eda/` | `ProcesadorEDA` | Estadística descriptiva, correlaciones y limpieza fina |
| `visualizacion/` | `Visualizador` | Gráficos (dispersión, series, heatmap, mapa, etc.) |
| `modelos/` | `ModeloML` | Entrenamiento y comparación de modelos de regresión |

## Fuentes de datos

- **Calidad del aire y clima:** API de [Open-Meteo](https://open-meteo.com/) (PM10, PM2.5, NO2, O3, temperatura, humedad, viento).
- **Tráfico:** datos de **congestión** por hora/día (relación tiempo con tráfico vs sin tráfico), por zona.
- Los datos ya limpios viven en `data/raw/*.csv`; el dataset unificado en `data/processed/df_final.csv`.

## Modelo

Predice **PM2.5 (µg/m³)** mediante regresión. Compara **Regresión Lineal**,
**KNN** y **Random Forest**, y elige el mejor según RMSE (en las pruebas, el
mejor suele ser **Random Forest**). Variables de entrada: hora, día de la
semana, meteorología, otros contaminantes y **nivel de congestión**.

## Notas

- El **tráfico** se representa con la **congestión real** (`congestion_ajustada`),
  no con un flujo vehicular inventado. **No se usan datos sintéticos.**
- La base de datos `*.db` **no se versiona** (está en `.gitignore`); se
  **regenera** al correr `python src/main.py`.
- El dashboard tiene un respaldo con datos DEMO por si no encuentra los reales.

## Equipo

| Rol | Módulos |
|---|---|
| Persona A | `datos/`, `api/` (adquisición y limpieza de fuentes) |
| Persona B | `basedatos/`, `eda/` (almacenamiento y exploración) |
| Persona C | `modelos/`, `visualizacion/` (ML y dashboard) |
