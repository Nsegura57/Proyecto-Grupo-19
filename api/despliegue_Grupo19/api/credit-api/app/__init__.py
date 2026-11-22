from pathlib import Path

with open(Path(__file__).resolve().parent / "VERSION") as f:
    __version__ = f.read().strip()
