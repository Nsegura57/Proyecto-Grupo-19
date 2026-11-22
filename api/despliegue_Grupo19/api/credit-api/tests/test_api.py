from app.config import settings

P = settings.API_V1_STR


def test_home(client):
    r = client.get("/")
    assert r.status_code == 200 and "Grupo 19" in r.text


def test_health(client):
    r = client.get(f"{P}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "1.0.0" and body["api_version"] == "1.0.0"


def test_model_info(client):
    r = client.get(f"{P}/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["umbral"] == 0.675
    assert abs(info["metricas_prueba"]["pr_auc"] - 0.4576) < 5e-4
    assert "reference_probs" not in info


def test_predict_one(client, credito_ejemplo):
    r = client.post(f"{P}/predict", json={"inputs": [credito_ejemplo]})
    assert r.status_code == 200
    b = r.json()
    assert len(b["predictions"]) == 1 and 0 <= b["predictions"][0] <= 1
    assert b["labels"][0] in (0, 1) and b["priority"][0] in ("Alta", "Media", "Baja")
    assert 0 < b["top_percent"][0] <= 100 and b["threshold"] == 0.675


def test_predict_optional_missing(client):
    c = {"P2097": 8, "P2098": 9, "P2106": 4, "P2099S2": 2022, "P2104": 1_500_000}
    r = client.post(f"{P}/predict", json={"inputs": [c]})
    assert r.status_code == 200


def test_predict_invalid_code(client, credito_ejemplo):
    r = client.post(f"{P}/predict", json={"inputs": [{**credito_ejemplo, "P2097": 99}]})
    assert r.status_code == 422


def test_predict_malformed_body(client):
    assert client.post(f"{P}/predict", json={"x": 1}).status_code == 422


def test_prioritize_orders_descending(client, credito_ejemplo):
    lote = [credito_ejemplo,
            {**credito_ejemplo, "P2099S2": 2022, "P2097": 8},
            {**credito_ejemplo, "P2099S2": 2005, "P2097": 4}]
    r = client.post(f"{P}/priorizar", json={"inputs": lote})
    assert r.status_code == 200
    b = r.json()
    probs = [x["probabilidad"] for x in b["ranking"]]
    assert b["total"] == 3 and probs == sorted(probs, reverse=True)
    assert sorted(x["indice"] for x in b["ranking"]) == [0, 1, 2]
    assert [x["posicion"] for x in b["ranking"]] == [1, 2, 3]


def test_batch_limit(client, credito_ejemplo, monkeypatch):
    monkeypatch.setattr(settings, "MAX_BATCH", 2)
    r = client.post(f"{P}/predict", json={"inputs": [credito_ejemplo] * 3})
    assert r.status_code == 413
