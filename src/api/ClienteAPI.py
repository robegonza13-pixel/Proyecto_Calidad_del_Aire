import requests
import pandas as pd

class ClienteAPI:
    def __init__(self, key_google=None):
        self.__key_google = key_google

    @staticmethod
    def __hacer_peticion(url, metodo="GET", params=None, headers=None, json_body=None):
        try:
            response = requests.request(metodo, url, params=params, headers=headers, json=json_body)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error codigo: {response.status_code}")
                print(f"Respuesta de Google: {response.text}")
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
        datos = ClienteAPI.__hacer_peticion(url, metodo="GET", params=params)
        if datos is None:
            return pd.DataFrame()

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
        datos = ClienteAPI.__hacer_peticion(url, metodo="GET", params=params)
        if datos is None:
            return pd.DataFrame()

        df = pd.DataFrame(datos["hourly"])
        df["latitude"] = latitude
        df["longitude"] = longitude
        return df

    def obtener_congestion(self, origen_lat, origen_lon, destino_lat, destino_lon, departure_time=None):
        url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.__key_google,
            "X-Goog-FieldMask": "routes.duration,routes.staticDuration"
        }
        payload = {
            "origin": {"location": {"latLng": {"latitude": origen_lat, "longitude": origen_lon}}},
            "destination": {"location": {"latLng": {"latitude": destino_lat, "longitude": destino_lon}}},
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE_OPTIMAL"
        }
        if departure_time is not None:
            payload["departureTime"] = departure_time

        datos = ClienteAPI.__hacer_peticion(url, metodo="POST", headers=headers, json_body=payload)

        if not datos or "routes" not in datos or not datos["routes"]:
            return pd.DataFrame()


        ruta = datos["routes"][0]
        duracion_trafico = float(ruta["duration"].rstrip("s"))
        duracion_normal = float(ruta["staticDuration"].rstrip("s"))

        if duracion_normal <= 0:
            return pd.DataFrame()
        ratio = duracion_trafico / duracion_normal

        return pd.DataFrame([{
            "origen_lat": origen_lat,
            "origen_lon": origen_lon,
            "destino_lat": destino_lat,
            "destino_lon": destino_lon,
            "departure_time": departure_time,
            "duracion_normal_seg": duracion_normal,
            "duracion_trafico_seg": duracion_trafico,
            "congestion_ratio": ratio
        }])

