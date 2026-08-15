"""
src/modelos/modelo_ml.py

Entrena y evalúa modelos supervisados de REGRESIÓN para predecir la
concentración de PM2.5 (µg/m³)

Modelos incluidos: Regresión Lineal, KNN Regressor
y Random Forest Regressor

Nota sobre "regresión múltiple":
En este proyecto no es un algoritmo aparte ni requiere una clase
distinta. sklearn.linear_model.LinearRegression ya calcula regresión
lineal MÚLTIPLE automáticamente en cuanto se le pasa más de una
variable predictora (X con más de una columna). Es decir: entrenar
LinearRegression con ['flujo_vehicular', 'temperatura', 'humedad', ...]
como predictoras de 'pm25' YA ES la regresión múltiple.
Si se usara una sola predictora, sería regresión lineal
simple. Es el mismo modelo; lo que cambia es cuántas columnas de X se
le pasan.

Este módulo usa el contrato de columnas de helpers/columnas.py, así que
NO asume nombres de columna fijos: usa `column_mapping` si el Dataframe
trae nombres distintos a los definidos ahí.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from helpers.columnas import CONTAMINANTES, METEOROLOGICAS, obtener_columna


class ModeloML:
    """
    Entrena, evalúa y compara modelos de regresión para predecir PM2.5.

    Parameters
    ----------
    df : pd.DataFrame
        Datos ya limpios (reales o de ejemplo/DEMO).
    variable_objetivo : str
        Nombre SEMÁNTICO de la variable a predecir (por defecto 'pm25').
    column_mapping : dict, opcional
        Igual que en Visualizador: permite remapear nombres semánticos
        ("pm25", "flujo_vehicular", ...) a los nombres reales de columna
        del DataFrame final, sin modificar esta clase. Ver
        helpers/columnas.py.

    Ejemplo
    -------
    >>> ml = ModeloML(df, variable_objetivo="pm25")
    >>> ml.seleccionar_variables()              # 2
    >>> ml.separar_x_y()                        # 4
    >>> ml.dividir_entrenamiento_prueba()       # 5
    >>> ml.entrenar_modelos()                   # 6
    >>> tabla = ml.evaluar_modelos()            # 7 y 8
    >>> ml.elegir_mejor_modelo()                # 9 y 10
    """

    # Modelos disponibles y sus hiperparámetros por defecto. Se usan
    # lambdas para que cada llamada a entrenar_modelos() cree instancias
    # nuevas (sin reutilizar un modelo ya entrenado de una corrida previa).
    MODELOS_DISPONIBLES = {
        "regresion_lineal": lambda: LinearRegression(),
        "knn": lambda: KNeighborsRegressor(n_neighbors=5),
        "random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    }

    def __init__(self, df: pd.DataFrame, variable_objetivo: str = "pm25", column_mapping: dict | None = None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("ModeloML espera un pandas DataFrame.")

        self.df = df.copy()
        self.variable_objetivo = variable_objetivo
        self.mapping = column_mapping or {}

        self.variables_predictoras: list[str] | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_esc = self.X_test_esc = None
        self.scaler: StandardScaler | None = None

        self.modelos_entrenados: dict[str, object] = {}
        self.resultados: dict[str, dict[str, float]] = {}
        self.mejor_modelo_nombre: str | None = None


    def _col(self, nombre_semantico: str) -> str:
        return obtener_columna(nombre_semantico, self.mapping)


    # 2. Seleccionar variables predictoras

    def seleccionar_variables(self, variables_predictoras: list[str] | None = None) -> list[str]:
        """
        Define qué columnas semánticas se usan como predictoras.

        Si no se especifican, se toman por defecto las variables que
        lista el proyecto (flujo vehicular, meteorología, hora, día de
        semana y los demás contaminantes distintos al objetivo) — solo
        las que ya existan en el DataFrame actual.
        """
        if variables_predictoras is not None:
            self.variables_predictoras = variables_predictoras
            return self.variables_predictoras

        candidatas = (
            ["flujo_vehicular"]
            + METEOROLOGICAS
            + ["hora", "dia_semana"]
            + [c for c in CONTAMINANTES if c != self.variable_objetivo]
        )
        self.variables_predictoras = [c for c in candidatas if self._col(c) in self.df.columns]

        if not self.variables_predictoras:
            raise ValueError(
                "No se encontró ninguna variable predictora disponible en el DataFrame. "
                "Revisa column_mapping o si el DataFrame ya tiene las columnas esperadas."
            )
        return self.variables_predictoras


    # 3 y 4. Variable objetivo (PM2.5) + separar X e y

    def separar_x_y(self) -> tuple[pd.DataFrame, pd.Series]:
        if self.variables_predictoras is None:
            self.seleccionar_variables()

        col_objetivo = self._col(self.variable_objetivo)
        if col_objetivo not in self.df.columns:
            raise ValueError(
                f"La variable objetivo '{self.variable_objetivo}' (columna '{col_objetivo}') "
                "no está en el DataFrame."
            )
        if not pd.api.types.is_numeric_dtype(self.df[col_objetivo]):
            raise ValueError(
                f"La variable objetivo '{self.variable_objetivo}' (columna '{col_objetivo}') no es "
                "numérica. ModeloML implementa REGRESIÓN, así que el objetivo debe ser un número "
                "(p. ej. pm25 en µg/m³), no una categoría."
            )

        cols_predictoras = [self._col(v) for v in self.variables_predictoras]

        # Las predictoras deben ser numéricas. Si alguna llega como texto
        # se avisa explícitamente en vez de transformarla "en silencio".
        no_numericas = [c for c in cols_predictoras if not pd.api.types.is_numeric_dtype(self.df[c])]
        if no_numericas:
            raise ValueError(
                f"Estas columnas predictoras no son numéricas: {no_numericas}. "
                "Conviértelas a número antes de usarlas, o quítalas de variables_predictoras."
            )

        datos = self.df[cols_predictoras + [col_objetivo]].dropna()
        if len(datos) < len(self.df):
            print(f"[separar_x_y] ℹ️  Se descartaron {len(self.df) - len(datos)} filas con datos faltantes.")

        self.X = datos[cols_predictoras]
        self.y = datos[col_objetivo]
        return self.X, self.y


    # 5. Dividir entrenamiento y prueba

    def dividir_entrenamiento_prueba(self, test_size: float = 0.2, random_state: int = 42):
        if self.X is None or self.y is None:
            self.separar_x_y()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Escalado: no afecta a Random Forest, pero sí importa para KNN
        # (que usa distancias entre puntos) y no perjudica a la regresión
        # lineal. El scaler se ajusta (fit) SOLO con entrenamiento para no
        # filtrar información del set de prueba.
        self.scaler = StandardScaler()
        self.X_train_esc = pd.DataFrame(
            self.scaler.fit_transform(self.X_train), columns=self.X_train.columns, index=self.X_train.index
        )
        self.X_test_esc = pd.DataFrame(
            self.scaler.transform(self.X_test), columns=self.X_test.columns, index=self.X_test.index
        )
        return self.X_train, self.X_test, self.y_train, self.y_test


    # 6. Entrenar los modelos

    def entrenar_modelos(self, modelos: list[str] | None = None) -> dict[str, object]:
        """
        Entrena los modelos indicados (por defecto, los 3 disponibles:
        'regresion_lineal', 'knn', 'random_forest').
        """
        if self.X_train is None:
            self.dividir_entrenamiento_prueba()

        modelos = modelos or list(self.MODELOS_DISPONIBLES.keys())
        for nombre in modelos:
            if nombre not in self.MODELOS_DISPONIBLES:
                raise ValueError(f"Modelo '{nombre}' no reconocido. Opciones: {list(self.MODELOS_DISPONIBLES)}")
            modelo = self.MODELOS_DISPONIBLES[nombre]()
            modelo.fit(self.X_train_esc, self.y_train)
            self.modelos_entrenados[nombre] = modelo

        return self.modelos_entrenados


    # 7. Generar predicciones

    def predecir(self, nombre_modelo: str, X_nuevo: pd.DataFrame | None = None) -> np.ndarray:
        """
        Predice con el modelo indicado. Si no se pasa `X_nuevo`, predice
        sobre el set de prueba (X_test).

        Si se pasa `X_nuevo`, sus columnas deben llamarse igual que las de
        `self.X_train` (es decir, los nombres REALES del DataFrame, no
        necesariamente los nombres semánticos de variables_predictoras).
        Para construir esa fila a partir de nombres semánticos, traduce
        cada uno con `obtener_columna(...)` y el mismo `column_mapping`:

            fila = {
                obtener_columna(var, ml.mapping): valor
                for var, valor in {"flujo_vehicular": 500, "viento": 4.0, ...}.items()
            }
            X_nuevo = pd.DataFrame([fila])
            ml.predecir(ml.mejor_modelo_nombre, X_nuevo)
        """
        if nombre_modelo not in self.modelos_entrenados:
            raise ValueError(f"El modelo '{nombre_modelo}' todavía no ha sido entrenado.")
        modelo = self.modelos_entrenados[nombre_modelo]

        if X_nuevo is None:
            X_usar = self.X_test_esc
        else:
            columnas_esperadas = list(self.X_train.columns)
            faltantes = [c for c in columnas_esperadas if c not in X_nuevo.columns]
            if faltantes:
                raise ValueError(f"Faltan columnas en X_nuevo para predecir: {faltantes}")
            X_usar = pd.DataFrame(
                self.scaler.transform(X_nuevo[columnas_esperadas]),
                columns=columnas_esperadas,
                index=X_nuevo.index,
            )

        return modelo.predict(X_usar)


    # 8. Calcular MAE, MSE, RMSE, R²

    def evaluar_modelos(self) -> pd.DataFrame:
        if not self.modelos_entrenados:
            raise ValueError("Todavía no hay modelos entrenados. Llama primero a entrenar_modelos().")

        for nombre in self.modelos_entrenados:
            y_pred = self.predecir(nombre)
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(self.y_test, y_pred)
            self.resultados[nombre] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

        return self.comparar_modelos()


    # 9. Comparar los modelos

    def comparar_modelos(self, metrica_orden: str = "RMSE") -> pd.DataFrame:
        if not self.resultados:
            raise ValueError("Todavía no hay resultados. Llama primero a evaluar_modelos().")

        tabla = pd.DataFrame(self.resultados).T
        ascendente = metrica_orden.upper() != "R2"  # para R2, mayor es mejor; para el resto, menor es mejor
        return tabla.sort_values(metrica_orden.upper(), ascending=ascendente)


    # 10. Identificar el mejor modelo

    def elegir_mejor_modelo(self, metrica: str = "RMSE") -> tuple[str, object]:
        tabla = self.comparar_modelos(metrica)
        self.mejor_modelo_nombre = tabla.index[0]
        return self.mejor_modelo_nombre, self.modelos_entrenados[self.mejor_modelo_nombre]

    def guardar_modelo(self, ruta: str = "modelo.pkl") -> None:
        """
        Guarda el mejor modelo entrenado junto con el scaler y las
        variables predictoras necesarias para reutilizarlo posteriormente.

        El archivo generado contiene:
        - modelo: modelo de Machine Learning entrenado.
        - scaler: StandardScaler utilizado durante el entrenamiento.
        - variables_predictoras: variables utilizadas por el modelo.
        - variable_objetivo: variable que el modelo predice.
        """

        if self.mejor_modelo_nombre is None:
            raise ValueError(
                "Todavía no se ha seleccionado el mejor modelo. "
                "Llama primero a elegir_mejor_modelo()."
            )

        if self.mejor_modelo_nombre not in self.modelos_entrenados:
            raise ValueError(
                f"El modelo '{self.mejor_modelo_nombre}' no está entrenado."
            )

        datos_modelo = {
            "modelo": self.modelos_entrenados[self.mejor_modelo_nombre],
            "scaler": self.scaler,
            "variables_predictoras": self.variables_predictoras,
            "variable_objetivo": self.variable_objetivo,
            "nombre_modelo": self.mejor_modelo_nombre,
        }

        joblib.dump(datos_modelo, ruta)

        print(f"Modelo guardado correctamente en: {ruta}")


    # Atajo opcional: corre los pasos 2-10 en una sola llamada

    def entrenar_y_evaluar(
        self, variables_predictoras: list[str] | None = None, modelos: list[str] | None = None
    ) -> pd.DataFrame:
        """Ejecuta seleccionar_variables → ... → elegir_mejor_modelo de una vez."""
        self.seleccionar_variables(variables_predictoras)
        self.separar_x_y()
        self.dividir_entrenamiento_prueba()
        self.entrenar_modelos(modelos)
        tabla = self.evaluar_modelos()
        self.elegir_mejor_modelo()
        return tabla


if __name__ == "__main__":
    # Prueba end-to-end con datos SINTÉTICOS (helpers/datos_demo.py).
    # Esto NO se ejecuta cuando la clase se importa desde otro módulo
    # (por ejemplo, desde el dashboard con el DataFrame real).
    from helpers.datos_demo import generar_datos_demo

    print("=" * 70)
    print("PRUEBA con datos SINTÉTICOS/DEMO (helpers/datos_demo.py)")
    print("Estos NO son datos reales de calidad del aire ni de tránsito del GAM.")
    print("Solo sirven para confirmar que el pipeline de ModeloML funciona.")
    print("=" * 70)

    df_demo = generar_datos_demo()
    ml = ModeloML(df_demo, variable_objetivo="pm25")
    tabla_resultados = ml.entrenar_y_evaluar()

    print(f"\nVariables predictoras usadas: {ml.variables_predictoras}")
    print(f"Filas de entrenamiento: {len(ml.X_train)} | Filas de prueba: {len(ml.X_test)}")
    print("\nComparación de modelos (métricas sobre datos DEMO, SIN valor real):")
    print(tabla_resultados.round(3))
    print(f"\nMejor modelo según RMSE (en datos DEMO): {ml.mejor_modelo_nombre}")
    print("\nListo: el pipeline corrió sin errores, de punta a punta, sobre datos de ejemplo.")