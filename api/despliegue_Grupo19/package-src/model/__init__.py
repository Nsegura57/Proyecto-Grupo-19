import logging
from pathlib import Path

logging.getLogger("model").addHandler(logging.NullHandler())

with open(Path(__file__).resolve().parent / "VERSION") as _f:
    __version__ = _f.read().strip()
