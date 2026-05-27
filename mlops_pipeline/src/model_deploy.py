"""
model_deploy.py
---------------
API de despliegue del modelo de predicción de riesgo crediticio.

Expone el modelo Gradient Boosting serializado en 'models/best_model.joblib'
como servicio REST mediante FastAPI. Soporta:
  - Predicción individual y por lotes vía JSON  (POST /predict)
  - Predicción por lotes vía archivo CSV         (POST /predict/csv)
  - Health check                                  (GET  /health)
  - Información del modelo                        (GET  /)
"""

import io
import os
import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")

try:
    artifact = joblib.load(MODEL_PATH)
    logger.info("Tipo de artifact cargado: %s", type(artifact))

    if isinstance(artifact, dict):
        logger.info("Claves disponibles en el artifact: %s", list(artifact.keys()))

        # Buscar el modelo probando nombres comunes
        MODEL_KEYS = ["model", "pipeline", "estimator", "classifier", "best_model"]
        MODEL = None
        for key in MODEL_KEYS:
            if key in artifact:
                MODEL = artifact[key]
                logger.info("Modelo encontrado con clave: '%s'", key)
                break

        # Si ninguna clave conocida funciona, tomar el primer valor que no sea numérico
        if MODEL is None:
            for key, val in artifact.items():
                if hasattr(val, "predict_proba"):
                    MODEL = val
                    logger.info("Modelo encontrado con clave alternativa: '%s'", key)
                    break

        if MODEL is None:
            raise ValueError(f"No se pudo identificar el modelo en el dict. Claves: {list(artifact.keys())}")

        # Buscar el threshold: la variable de entorno siempre tiene prioridad
        env_threshold = os.getenv("PREDICT_THRESHOLD")
        if env_threshold is not None:
            THRESHOLD = float(env_threshold)
            logger.info("Threshold tomado de variable de entorno PREDICT_THRESHOLD: %.4f", THRESHOLD)
        else:
            THRESHOLD_KEYS = ["threshold", "best_threshold", "optimal_threshold", "umbral"]
            THRESHOLD = 0.5
            for key in THRESHOLD_KEYS:
                if key in artifact:
                    THRESHOLD = artifact[key]
                    logger.info("Threshold tomado del modelo guardado, clave '%s': %.4f", key, THRESHOLD)
                    break
    else:
        # El artifact es directamente el modelo (Pipeline o estimador)
        MODEL = artifact
        THRESHOLD = float(os.getenv("PREDICT_THRESHOLD", "0.5"))
        logger.info("Artifact es el modelo directamente (no dict)")

    logger.info("Modelo cargado correctamente desde '%s'", MODEL_PATH)
    logger.info("Threshold de clasificación: %.4f", THRESHOLD)
except FileNotFoundError:
    logger.error("No se encontró el modelo en '%s'", MODEL_PATH)
    MODEL = None
    THRESHOLD = 0.5
except Exception as exc:
    logger.error("Error al cargar el modelo: %s", exc)
    MODEL = None
    THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Extraer las features esperadas directamente del modelo cargado
# Así siempre estarán sincronizadas con lo que se usó en el entrenamiento
# ---------------------------------------------------------------------------
def _extract_feature_names(model) -> Optional[List[str]]:
    """Intenta extraer los nombres de features del pipeline entrenado."""
    if model is None:
        return None
    # Caso 1: el objeto tiene feature_names_in_ directamente
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # Caso 2: es un Pipeline de sklearn → buscar en sus steps
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
            # Caso 3: el step es un ColumnTransformer
            if hasattr(step, "transformers_"):
                cols = []
                for _, _, columns in step.transformers_:
                    if isinstance(columns, list):
                        cols.extend(columns)
                if cols:
                    return cols
    return None

FEATURE_COLUMNS = _extract_feature_names(MODEL)

if FEATURE_COLUMNS:
    logger.info("Features extraídas del modelo (%d): %s", len(FEATURE_COLUMNS), FEATURE_COLUMNS)
else:
    logger.warning("No se pudieron extraer features del modelo. Se aceptarán todos los campos enviados.")

# ---------------------------------------------------------------------------
# Mapeo ordinal para tendencia_ingresos
# El OrdinalEncoder del pipeline entrenado convirtió los strings a números.
# La API acepta tanto el string como el número para mayor comodidad.
# ---------------------------------------------------------------------------
TENDENCIA_INGRESOS_MAP = {
    "Decreciente":   0,
    "Sin_historial": 1,
    "Estable":       2,
    "Creciente":     3,
}

# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------

class BatchRequest(BaseModel):
    """
    Solicitud de predicción por lotes (JSON).
    Cada registro es un dict con los campos del modelo.
    Usá GET /features para ver qué campos se esperan.
    """
    records: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de registros a predecir. Cada registro es un objeto con los campos del modelo.",
        examples=[[{"cant_creditosvigentes": 2, "capital_prestado": 5000000, "tendencia_ingresos": "Estable"}]]
    )


class PredictionResult(BaseModel):
    """Resultado de predicción para un registro."""
    record_index: int
    probability_mora: float = Field(..., description="Probabilidad de caer en mora (clase 0)")
    prediction: int = Field(..., description="Predicción binaria: 0=Mora, 1=Pago a tiempo")
    risk_label: str = Field(..., description="Etiqueta de riesgo: MORA o PAGO_A_TIEMPO")


class BatchResponse(BaseModel):
    """Respuesta completa de predicción por lotes."""
    total_records: int
    threshold_used: float
    predictions: List[PredictionResult]
    summary: dict


# ---------------------------------------------------------------------------
# Inicialización de FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Predicción de Riesgo Crediticio",
    description=(
        "Modelo de clasificación binaria que predice si un cliente pagará a tiempo "
        "(Pago_atiempo=1) o caerá en mora (Pago_atiempo=0). "
        "Modelo: Gradient Boosting Classifier | Métrica principal: ROC-AUC"
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Función auxiliar
# ---------------------------------------------------------------------------

def _predict_dataframe(df: pd.DataFrame) -> BatchResponse:
    """Ejecuta predicciones sobre un DataFrame y retorna la respuesta formateada."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Verifique que 'models/best_model.joblib' existe.",
        )

    # Convertir tendencia_ingresos de string a número si viene como texto
    if "tendencia_ingresos" in df.columns:
        df["tendencia_ingresos"] = df["tendencia_ingresos"].apply(
            lambda x: TENDENCIA_INGRESOS_MAP.get(x, x) if isinstance(x, str) else x
        )

    # Si conocemos las features del modelo, alinear el DataFrame a ellas
    if FEATURE_COLUMNS:
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[FEATURE_COLUMNS]
    # Si no las conocemos, pasar el df tal cual (el pipeline interno se encarga)

    try:
        proba = MODEL.predict_proba(df)[:, 0]  # probabilidad de mora (clase 0)
        predictions = (proba >= THRESHOLD).astype(int)
        # Si prob_mora >= threshold → predice mora (0); si no → pago a tiempo (1)
        binary_pred = np.where(proba >= THRESHOLD, 0, 1)
    except Exception as exc:
        logger.error("Error durante la predicción: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(exc)}")

    results = []
    for i, (prob, pred) in enumerate(zip(proba, binary_pred)):
        results.append(
            PredictionResult(
                record_index=i,
                probability_mora=round(float(prob), 6),
                prediction=int(pred),
                risk_label="MORA" if pred == 0 else "PAGO_A_TIEMPO",
            )
        )

    n_mora = int(np.sum(binary_pred == 0))
    n_pago = int(np.sum(binary_pred == 1))

    return BatchResponse(
        total_records=len(results),
        threshold_used=THRESHOLD,
        predictions=results,
        summary={
            "predicciones_mora": n_mora,
            "predicciones_pago_a_tiempo": n_pago,
            "tasa_mora_estimada": round(n_mora / len(results), 4) if results else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    """Información general del servicio."""
    return {
        "servicio": "API de Predicción de Riesgo Crediticio",
        "version": "1.0.0",
        "modelo": "Gradient Boosting Classifier",
        "metrica_principal": "ROC-AUC = 0.677 (Gini = 0.353)",
        "threshold": THRESHOLD,
        "endpoints": {
            "GET  /health": "Estado del servicio",
            "POST /predict": "Predicción por lotes vía JSON",
            "POST /predict/csv": "Predicción por lotes vía archivo CSV",
            "GET  /docs": "Documentación interactiva (Swagger UI)",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check del servicio."""
    model_status = "ok" if MODEL is not None else "error - modelo no cargado"
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "modelo_cargado": MODEL is not None,
        "modelo_status": model_status,
        "threshold": THRESHOLD,
    }


@app.get("/features", tags=["Info"])
def features():
    """Lista las features que el modelo espera recibir."""
    if FEATURE_COLUMNS is None:
        return {"features": None, "mensaje": "No se pudieron extraer las features del modelo."}
    return {
        "total_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "nota": "Podés omitir campos; el pipeline los imputará con la mediana o la moda según el tipo.",
    }


@app.post("/predict", response_model=BatchResponse, tags=["Predicción"])
def predict_json(batch: BatchRequest):
    """
    Predicción por lotes a partir de un JSON.

    Enviar una lista de registros en el campo `records`.
    Cada registro puede tener valores nulos; el pipeline interno los imputa.
    Retorna la probabilidad de mora, la predicción binaria y un resumen del lote.
    """
    logger.info("POST /predict — %d registros recibidos", len(batch.records))
    df = pd.DataFrame(batch.records)
    return _predict_dataframe(df)


@app.post("/predict/csv", response_model=BatchResponse, tags=["Predicción"])
async def predict_csv(file: UploadFile = File(..., description="Archivo CSV con los registros a predecir")):
    """
    Predicción por lotes a partir de un archivo CSV.

    El CSV debe tener encabezados con los nombres de las variables.
    Las columnas no presentes se imputarán automáticamente.
    Retorna la misma estructura que POST /predict.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe tener extensión .csv")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo parsear el CSV: {str(exc)}")

    logger.info("POST /predict/csv — archivo '%s', %d filas", file.filename, len(df))

    # Eliminar columnas target si vienen en el CSV (evitar leakage accidental)
    for col in ["Pago_atiempo", "pago_atiempo", "saldo_mora", "saldo_mora_codeudor", "puntaje"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return _predict_dataframe(df)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "model_deploy:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )