from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from model.config.core import config


class DataInputSchema(BaseModel):
    """Un crédito: las 7 variables del modelo (códigos de ELCO 2022)."""

    P2097: int = Field(ge=1, le=15, description="Entidad acreedora (código 1-15)")
    P2098: int = Field(ge=1, le=16, description="Destino del crédito (código 1-16)")
    P2106: int = Field(ge=1, le=11, description="Periodicidad de pago (código 1-11)")
    P2099S2: int = Field(ge=1950, le=2030, description="Año de inicio del crédito")
    P2104: float = Field(ge=0, description="Valor inicial del crédito (pesos)")
    P2105S1: Optional[float] = Field(default=None, ge=0, description="Valor de la cuota (pesos)")
    P2107S1: Optional[float] = Field(default=None, ge=0, description="Plazo en meses (vacío o 9999 = sin plazo definido)")


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Revisa que las entradas sean procesables. No elimina filas: la salida conserva el orden de entrada."""
    features = config.model_cfg.features
    input_data = input_data.copy()
    for nombre, campo in DataInputSchema.model_fields.items():     # las variables opcionales pueden omitirse
        if not campo.is_required() and nombre not in input_data.columns:
            input_data[nombre] = np.nan
    faltan = [c for c in features if c not in input_data.columns]
    if faltan:
        return input_data, f'[{{"type": "missing_columns", "msg": "Faltan columnas: {faltan}"}}]'

    relevant = input_data[features].copy()
    errors = None
    try:
        MultipleDataInputs(inputs=relevant.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()
    return relevant, errors
