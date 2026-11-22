#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Empaquetamiento del modelo de priorización de créditos (Grupo 19, ELCO 2022)."""
from pathlib import Path

from setuptools import find_packages, setup

NAME = "model-atrasos"
DESCRIPTION = "Modelo Gradient Boosting (variables del crédito) para priorizar créditos por posibles retrasos."
URL = "https://github.com/Nsegura57/Proyecto-Grupo-19"
AUTHOR = "Grupo 19"
REQUIRES_PYTHON = ">=3.12.0"

ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR = ROOT_DIR / "model"
with open(PACKAGE_DIR / "VERSION") as f:
    VERSION = f.read().strip()


def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return [l for l in fd.read().splitlines() if l and not l.startswith("#")]


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"model": ["VERSION", "config.yml", "datasets/*.csv", "trained/*.pkl", "trained/*.json"]},
    install_requires=list_reqs(),
    include_package_data=True,
    license="BSD-3",
    classifiers=["Programming Language :: Python :: 3"],
)
