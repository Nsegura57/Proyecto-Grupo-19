import pytest

from model.config.core import config
from model.processing.data_manager import load_dataset, prepare_features, split_partitions


@pytest.fixture(scope="session")
def test_partition():
    """Partición de prueba por hogar (592 créditos, 95 con retrasos)."""
    data = prepare_features(load_dataset(file_name=config.app_config.train_data_file))
    _, _, test = split_partitions(data)
    return test


@pytest.fixture()
def sample_input_data(test_partition):
    return test_partition[config.model_cfg.features]
