import hashlib

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from model import __version__ as _version
from model.config.core import DATASET_DIR, config
from model.pipeline import build_pipeline
from model.processing.data_manager import (
    load_dataset,
    prepare_features,
    save_card,
    save_pipeline,
    split_partitions,
)

UMBRALES = np.round(np.arange(0.10, 0.901, 0.025), 3)


def metricas(y, p, thr):
    pred = (np.asarray(p) >= thr).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "umbral": float(thr),
    }


def priorizacion(y, p):
    """Revisar el 10/20/30 % de créditos con mayor probabilidad: cuántos retrasos se encuentran."""
    y = np.asarray(y, dtype=int)
    ordenado = y[np.argsort(-np.asarray(p), kind="stable")]
    filas = []
    for frac in (0.10, 0.20, 0.30):
        n = max(1, int(np.ceil(len(y) * frac)))
        enc = int(ordenado[:n].sum())
        filas.append({"porcentaje_revisado": int(frac * 100), "creditos_revisados": n,
                      "retrasos_encontrados": enc, "precision_top": enc / n,
                      "recall_top": enc / int(y.sum()),
                      "creditos_por_retraso": (n / enc) if enc else None})
    return filas


def run_training() -> None:
    """Entrena el modelo con la partición de entrenamiento y guarda pipeline y ficha del modelo."""
    cfg = config.model_cfg
    data = prepare_features(load_dataset(file_name=config.app_config.train_data_file))
    train, val, test = split_partitions(data)
    cols = cfg.features

    pipe = build_pipeline()
    pipe.fit(train[cols], train[cfg.target])

    p_val = pipe.predict_proba(val[cols])[:, 1]
    p_test = pipe.predict_proba(test[cols])[:, 1]
    umbral_val = float(UMBRALES[int(np.argmax([f1_score(val[cfg.target], p_val >= u, zero_division=0)
                                               for u in UMBRALES]))])

    card = {
        "algoritmo": "HistGradientBoostingClassifier",
        "conjunto_variables": "A (características del crédito)",
        "version_paquete": _version,
        "scikit_learn": sklearn_version,
        "variables": cols,
        "variables_categoricas": cfg.categorical_vars,
        "hiperparametros": {"learning_rate": cfg.learning_rate, "max_iter": cfg.max_iter,
                            "max_leaf_nodes": cfg.max_leaf_nodes, "min_samples_leaf": cfg.min_samples_leaf,
                            "l2_regularization": cfg.l2_regularization, "class_weight": "balanced"},
        "umbral": cfg.threshold,
        "umbral_maximo_f1_validacion": umbral_val,
        "particiones": {"train": len(train), "val": len(val), "test": len(test)},
        "retrasos_prueba": int(test[cfg.target].sum()),
        "tasa_base_prueba": float(test[cfg.target].mean()),
        "metricas_validacion": metricas(val[cfg.target], p_val, cfg.threshold),
        "metricas_prueba": metricas(test[cfg.target], p_test, cfg.threshold),
        "priorizacion_prueba": priorizacion(test[cfg.target], p_test),
        "sha256_datos": hashlib.sha256((DATASET_DIR / config.app_config.train_data_file).read_bytes()).hexdigest(),
        "reference_probs": [round(float(x), 6) for x in p_val],
        "nota_referencia": "reference_probs: probabilidades del conjunto de validación (población de referencia).",
    }
    save_pipeline(pipeline_to_persist=pipe)
    save_card(card)
    print("Modelo entrenado. Umbral configurado:", cfg.threshold,
          "| umbral de máximo F1 en validación:", umbral_val)
    print("Métricas de prueba:", {k: round(v, 4) for k, v in card["metricas_prueba"].items()})


if __name__ == "__main__":
    run_training()
