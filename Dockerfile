# ---------------------------------------------------------------------------
# Dockerfile — MLOps Pipeline: Predicción de Riesgo Crediticio
# ---------------------------------------------------------------------------
# Imagen base: Python 3.10 slim para mantener el tamaño reducido
FROM python:3.11-slim

# Metadata
LABEL maintainer="caromponce@gmail.com"
LABEL description="API FastAPI para predicción de riesgo crediticio"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="models/best_model.joblib" \
    PORT=8000

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema (necesarias para scipy/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python primero (capa cacheada)
# Se filtran paquetes exclusivos de Windows que no existen en Linux
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && grep -v -iE "^(pywin32|pywinpty|pyreadline|colorama|wmi|winshell|win32)" requirements.txt \
       > /tmp/requirements_linux.txt \
    && pip install --no-cache-dir -r /tmp/requirements_linux.txt

# Copiar el código fuente
COPY mlops_pipeline/src/model_deploy.py ./mlops_pipeline/src/model_deploy.py

# Copiar el modelo serializado
COPY models/ ./models/

# Copiar recursos de monitoreo (estadísticas de referencia)
COPY monitoring/ ./monitoring/

# Exponer el puerto de la API
EXPOSE ${PORT}

# Health check del contenedor
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

# Comando de inicio: Uvicorn como servidor ASGI
CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]