import json
import typing as t

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / file_name)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza común a entrenamiento y predicción: 9999 = plazo sin respuesta -> vacío."""
    cfg = config.model_cfg
    out = df.copy()
    if "P2107S1" in out.columns:
        out["P2107S1"] = pd.to_numeric(out["P2107S1"], errors="coerce").replace(cfg.no_term_code, np.nan)
    for col in cfg.numeric_vars:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def split_partitions(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición por hogar 70/15/15, idéntica a la de los cuadernos del grupo."""
    cfg = config.model_cfg
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size_first, random_state=cfg.random_state)
    i_train, i_tmp = next(gss.split(df, df[cfg.target], groups=df[cfg.group]))
    train, tmp = df.iloc[i_train].copy(), df.iloc[i_tmp].copy()
    gss2 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size_second, random_state=cfg.random_state + 1)
    i_val, i_test = next(gss2.split(tmp, tmp[cfg.target], groups=tmp[cfg.group]))
    return train, tmp.iloc[i_val].copy(), tmp.iloc[i_test].copy()


def pipeline_file_name() -> str:
    return f"{config.app_config.pipeline_save_file}{_version}.pkl"


def card_file_name() -> str:
    return f"{config.app_config.card_save_file}{_version}.json"


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Guarda el pipeline versionado y borra versiones anteriores."""
    names = [pipeline_file_name(), card_file_name()]
    remove_old_pipelines(files_to_keep=names)
    joblib.dump(pipeline_to_persist, TRAINED_MODEL_DIR / pipeline_file_name())


def load_pipeline(*, file_name: str) -> Pipeline:
    return joblib.load(filename=TRAINED_MODEL_DIR / file_name)


def save_card(card: dict) -> None:
    (TRAINED_MODEL_DIR / card_file_name()).write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_card() -> dict:
    return json.loads((TRAINED_MODEL_DIR / card_file_name()).read_text(encoding="utf-8"))


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """Deja un solo modelo entrenado por versión del paquete."""
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
