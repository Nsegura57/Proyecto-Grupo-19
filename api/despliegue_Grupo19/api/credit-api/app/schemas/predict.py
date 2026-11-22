from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any] = None
    version: str
    threshold: float
    predictions: Optional[List[float]] = None   # probabilidad de retraso
    labels: Optional[List[int]] = None          # 1 si probabilidad >= umbral
    top_percent: Optional[List[float]] = None   # % de la población de referencia con mayor probabilidad
    priority: Optional[List[str]] = None        # Alta / Media / Baja


class CreditInput(BaseModel):
    """Crédito con las 7 variables del modelo (códigos de ELCO 2022)."""

    P2097: Optional[int] = None
    P2098: Optional[int] = None
    P2106: Optional[int] = None
    P2099S2: Optional[int] = None
    P2104: Optional[float] = None
    P2105S1: Optional[float] = None
    P2107S1: Optional[float] = None


class MultipleCreditInputs(BaseModel):
    inputs: List[CreditInput]

    model_config = {
        "json_schema_extra": {
            "example": {
                "inputs": [
                    {"P2097": 1, "P2098": 2, "P2106": 4, "P2099S2": 2018,
                     "P2104": 25000000, "P2105S1": 450000, "P2107S1": 60},
                    {"P2097": 8, "P2098": 9, "P2106": 4, "P2099S2": 2022,
                     "P2104": 1500000, "P2105S1": None, "P2107S1": None},
                ]
            }
        }
    }


class RankedCredit(BaseModel):
    posicion: int
    indice: int            # posición del crédito en la lista enviada (0 = primero)
    probabilidad: float
    etiqueta: int
    percentil_top: float
    prioridad: str
    grupo: str             # Top 10 % / 10–20 % / 20–30 % / Resto


class PrioritizationResults(BaseModel):
    errors: Optional[Any] = None
    version: str
    threshold: float
    total: int = 0
    ranking: Optional[List[RankedCredit]] = None
