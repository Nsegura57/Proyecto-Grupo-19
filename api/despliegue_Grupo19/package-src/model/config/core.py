from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from strictyaml import YAML, load

import model

PACKAGE_ROOT = Path(model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained"


class AppConfig(BaseModel):
    """Configuración a nivel de aplicación."""

    package_name: str
    train_data_file: str
    pipeline_save_file: str
    card_save_file: str


class ModelConfig(BaseModel):
    """Configuración del modelo y de la ingeniería de variables."""

    target: str
    group: str
    features: List[str]
    categorical_vars: List[str]
    numeric_vars: List[str]
    no_term_code: int
    random_state: int
    test_size_first: float
    test_size_second: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    threshold: float
    priority_top_high: float
    priority_top_medium: float


class Config(BaseModel):
    """Objeto maestro de configuración."""

    app_config: AppConfig
    model_cfg: ModelConfig


def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"No se encuentra la configuración en {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    if not cfg_path:
        cfg_path = find_config_file()
    with open(cfg_path, "r") as conf_file:
        return load(conf_file.read())


def create_and_validate_config(parsed_config: Optional[YAML] = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
    return Config(
        app_config=AppConfig(**parsed_config.data),
        model_cfg=ModelConfig(**parsed_config.data),
    )


config = create_and_validate_config()
