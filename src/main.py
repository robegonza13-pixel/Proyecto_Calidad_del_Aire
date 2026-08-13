from api.ClienteAPI import ClienteAPI

prueba_aire = ClienteAPI.obtener_calidad_aire(9.9358, -84.1043, "2026-08-11", "2026-08-11")
print(prueba_aire)
prueba_clima = ClienteAPI.obtener_clima(9.9358, -84.1043, "2026-08-11", "2026-08-11")
print(prueba_clima)