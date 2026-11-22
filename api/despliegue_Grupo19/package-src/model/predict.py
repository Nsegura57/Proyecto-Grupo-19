import typing as t

import numpy as np
import pandas as pd

from model import __version__ as _version
from model.config.core import config
from model.processing.data_manager import (
    card_file_name,
    load_card,
    load_pipeline,
    pipeline_file_name,
    prepare_features,
)
from model.processing.validation import validate_inputs

_pipe = load_pipeline(file_name=pipeline_file_name())
_card = load_card()
_reference = np.sort(np.asarray(_card["reference_probs"], dtype=float))


def top_percent(prob: float) -> float:
    """Porcentaje de la población de referencia con probabilidad igual o mayor (0-100)."""
    return float(100.0 * (len(_reference) - np.searchsorted(_reference, prob, side="left")) / len(_reference))


def priority_level(top_pct: float) -> str:
    cfg = config.model_cfg
    if top_pct <= cfg.priority_top_high:
        return "Alta"
    if top_pct <= cfg.priority_top_medium:
        return "Media"
    return "Baja"


def model_info() -> dict:
    """Resumen del modelo desplegado: variables, umbral y métricas de prueba."""
    return {k: v for k, v in _card.items() if k != "reference_probs"} | {"version": _version}


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict, list]) -> dict:
    """Probabilidad de retraso, clase según el umbral y prioridad de revisión de cada crédito."""
    data = pd.DataFrame(input_data)
    relevant, errors = validate_inputs(input_data=data)
    results: dict = {"predictions": None, "labels": None, "top_percent": None, "priority": None,
                     "threshold": config.model_cfg.threshold, "version": _version, "errors": errors}
    if errors:
        return results

    prepared = prepare_features(relevant)
    probs = _pipe.predict_proba(prepared[config.model_cfg.features])[:, 1]
    tops = [top_percent(float(p)) for p in probs]
    results.update(
        predictions=[float(p) for p in probs],
        labels=[int(p >= config.model_cfg.threshold) for p in probs],
        top_percent=tops,
        priority=[priority_level(x) for x in tops],
    )
    return results
