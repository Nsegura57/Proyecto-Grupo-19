from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model.config.core import config
from model.processing.features import CategoryToString


def build_pipeline() -> Pipeline:
    """Pipeline del modelo final: preparación de variables + Gradient Boosting."""
    cfg = config.model_cfg
    preparacion = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), cfg.numeric_vars),
            (
                "cat",
                Pipeline(
                    [
                        ("texto", CategoryToString()),
                        ("imputar", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cfg.categorical_vars,
            ),
        ]
    )
    modelo = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        class_weight="balanced",
        random_state=cfg.random_state,
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
    )
    return Pipeline([("preparacion", preparacion), ("modelo", modelo)])
