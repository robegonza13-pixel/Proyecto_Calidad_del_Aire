import requests
import pandas as pd

class ClienteAPI:
    def __init__(self, key_google=None):
        self.__key_google = key_google

    @staticmethod
    def __hacer_peticion(url, params=None, headers=None):
        try:
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error codigo: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"No se pudo conectar: {e}")
            return None

    @staticmethod
    def obtener_calidad_aire(latitude, longitude, start_date, end_date):
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "pm10,pm2_5,nitrogen_dioxide,ozone",
            "start_date": start_date,
            "end_date": end_date
        }
        datos = ClienteAPI.__hacer_peticion(url, params)
        if datos is None:
            return pd.DataFrame()
        else:
            df = pd.DataFrame(datos["hourly"])
            df["latitude"] = latitude
            df["longitude"] = longitude
            return df

    @staticmethod
    def obtener_clima(latitude, longitude, start_date, end_date):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "start_date": start_date,
            "end_date": end_date
        }
        datos = ClienteAPI.__hacer_peticion(url, params)
        if datos is None:
            return pd.DataFrame()
        else:
            df = pd.DataFrame(datos["hourly"])
            df["latitude"] = latitude
            df["longitude"] = longitude
            return df


