from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from modelo import run_analysis  # Asegúrate de que modelo.py esté en el mismo directorio

app = FastAPI()

# Montar la carpeta static para imágenes
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurar templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    resultados = run_analysis()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "correlation_matrix": resultados["correlation_matrix"],    # 🔢 Tabla de correlación
        "r2_scores": resultados["r2_scores"],                      # 📈 Diccionario con todos los R²
        "yearly_avg": resultados["yearly_avg"],                    # 📆 Promedio anual PM25
        "imagenes": {
            "heatmap": "static/heatmap_correlacion.png",                         # 🧮 Mapa de calor
            "dispersion_pm10": "static/dispersion_pm10.png",                    # 📉 Dispersión PM25 vs PM10
            "dispersion_pm10_vs_pm25": "static/dispersion_pm10_vs_pm25.png",    # ✅ Dispersión PM10 vs PM25
            "dispersion_pm25_multi": "static/dispersion_pm25_multivariada.png", # 📉 Dispersión multivariada PM25
            "dispersion_pm10_multi": "static/dispersion_pm10_multivariada.png", # 📉 Dispersión multivariada PM10
            "regresion_pm10": "static/regresion_pm10.png",                      # 📈 Regresión univariada PM10
            "regresion_pm25_multi": "static/regresion_pm25_multivariada.png",   # 📈 Regresión multivariada PM25
            "regresion_pm10_multi": "static/regresion_pm10_multivariada.png",   # 📈 Regresión multivariada PM10
            "pm25_anual": "static/pm25_anual.png",
            "dispersion_pm25_vs_pm10": "static/dispersion_pm25_vs_pm10.png"
    # 📊 Gráfico promedio anual PM25
        }
    })
