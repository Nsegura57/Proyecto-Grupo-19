import logging
import sys
from types import FrameType
from typing import List, cast

from loguru import logger
from pydantic_settings import BaseSettings


class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "API de priorización de créditos (Grupo 19)"
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    # Máximo de créditos por solicitud
    MAX_BATCH: int = 5000
    logging: LoggingSettings = LoggingSettings()


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_app_logging(config: "Settings") -> None:
    """Redirige los logs de uvicorn/fastapi a loguru."""
    logging.getLogger().handlers = [InterceptHandler()]
    for name in ("uvicorn", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [InterceptHandler()]
    logger.configure(handlers=[{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}])


settings = Settings()
