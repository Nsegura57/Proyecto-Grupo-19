import math

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from model.config.core import config
from model.predict import make_prediction, model_info


def test_make_prediction(sample_input_data):
    # Dado: los 592 créditos de la partición de prueba
    expected_no_predictions = 592
    expected_first_prediction_value = 0.9196

    # Cuando
    result = make_prediction(input_data=sample_input_data)

    # Entonces
    probs = result["predictions"]
    assert result["errors"] is None
    assert isinstance(probs, list) and len(probs) == expected_no_predictions
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert math.isclose(probs[0], expected_first_prediction_value, abs_tol=0.001)
    assert result["labels"] == [int(p >= config.model_cfg.threshold) for p in probs]
    assert set(result["priority"]) <= {"Alta", "Media", "Baja"}


def test_metrics_match_reported_values(sample_input_data, test_partition):
    """El modelo empaquetado reproduce las métricas de prueba reportadas por el equipo."""
    y = test_partition[config.model_cfg.target].to_numpy()
    probs = np.asarray(make_prediction(input_data=sample_input_data)["predictions"])
    pred = (probs >= config.model_cfg.threshold).astype(int)
    assert math.isclose(average_precision_score(y, probs), 0.4576, abs_tol=5e-4)
    assert math.isclose(precision_score(y, pred), 0.5111, abs_tol=5e-4)
    assert math.isclose(recall_score(y, pred), 0.4842, abs_tol=5e-4)
    assert math.isclose(f1_score(y, pred), 0.4973, abs_tol=5e-4)


def test_prioritization_top_10_percent(sample_input_data, test_partition):
    """Revisando el 10 % de mayor probabilidad (60 créditos) se encuentran 30 de los 95 retrasos."""
    y = test_partition[config.model_cfg.target].to_numpy()
    probs = np.asarray(make_prediction(input_data=sample_input_data)["predictions"])
    top = np.argsort(-probs, kind="stable")[:60]
    assert int(y[top].sum()) == 30


def test_invalid_category_returns_errors():
    bad = [{"P2097": 99, "P2098": 1, "P2106": 4, "P2099S2": 2020, "P2104": 1_000_000}]
    result = make_prediction(input_data=bad)
    assert result["errors"] is not None
    assert result["predictions"] is None


def test_optional_variables_can_be_missing():
    credit = [{"P2097": 1, "P2098": 2, "P2106": 4, "P2099S2": 2022, "P2104": 5_000_000}]
    result = make_prediction(input_data=credit)
    assert result["errors"] is None
    assert 0.0 <= result["predictions"][0] <= 1.0


def test_no_term_code_equals_missing_term():
    base = {"P2097": 1, "P2098": 2, "P2106": 4, "P2099S2": 2020, "P2104": 3_000_000, "P2105S1": 150_000}
    a = make_prediction(input_data=[{**base, "P2107S1": 9999}])["predictions"][0]
    b = make_prediction(input_data=[{**base, "P2107S1": None}])["predictions"][0]
    assert math.isclose(a, b, abs_tol=1e-12)


def test_model_info_has_reported_metrics():
    info = model_info()
    assert info["umbral"] == config.model_cfg.threshold
    assert math.isclose(info["metricas_prueba"]["pr_auc"], 0.4576, abs_tol=5e-4)
