import json
import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from model import __version__ as model_version
from model.predict import make_prediction, model_info

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """Estado de la API y versiones."""
    return schemas.Health(name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version).model_dump()


@api_router.get("/model-info", status_code=200)
def get_model_info() -> dict:
    """Ficha del modelo desplegado: variables, hiperparámetros, umbral y métricas de prueba."""
    return model_info()


def _to_frame(inputs: Any) -> pd.DataFrame:
    if len(inputs) > settings.MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Máximo {settings.MAX_BATCH} créditos por solicitud")
    df = pd.DataFrame(jsonable_encoder(inputs)).replace({np.nan: None})
    return df


def _clean_errors(errors):
    return json.loads(errors) if isinstance(errors, str) else errors


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleCreditInputs) -> Any:
    """Probabilidad de retraso, clase, posición en la población de referencia y prioridad de cada crédito."""
    df = _to_frame(input_data.inputs)
    logger.info(f"Predicción de {len(df)} crédito(s)")
    results = make_prediction(input_data=df)
    if results["errors"] is not None:
        logger.warning(f"Errores de validación: {results['errors']}")
        raise HTTPException(status_code=422, detail=_clean_errors(results["errors"]))
    return results


@api_router.post("/priorizar", response_model=schemas.PrioritizationResults, status_code=200)
async def prioritize(input_data: schemas.MultipleCreditInputs) -> Any:
    """Ordena un lote de créditos de mayor a menor probabilidad de retraso (cola de revisión)."""
    df = _to_frame(input_data.inputs)
    logger.info(f"Priorización de {len(df)} crédito(s)")
    r = make_prediction(input_data=df)
    if r["errors"] is not None:
        raise HTTPException(status_code=422, detail=_clean_errors(r["errors"]))

    probs = np.asarray(r["predictions"])
    n = len(probs)
    order = np.argsort(-probs, kind="stable")
    ranking = []
    for pos, idx in enumerate(order, start=1):
        share = 100.0 * pos / n
        grupo = "Top 10 %" if share <= 10 else "10–20 %" if share <= 20 else "20–30 %" if share <= 30 else "Resto"
        ranking.append(schemas.RankedCredit(
            posicion=pos, indice=int(idx), probabilidad=float(probs[idx]), etiqueta=r["labels"][idx],
            percentil_top=float(r["top_percent"][idx]), prioridad=r["priority"][idx], grupo=grupo))
    return schemas.PrioritizationResults(version=r["version"], threshold=r["threshold"], total=n, ranking=ranking)
