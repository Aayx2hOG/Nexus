# Nexus

A network intrusion detection research project exploring known-attack classification, novel-attack detection, calibration, and explainable alerts.

## Status

Initial Python project scaffold only. Data processing, modelling, and evaluation are not implemented yet.

## Local setup

Requires Python 3.11 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Python code lives in `src/nexus/`. Keep local datasets in `data/` and generated outputs in `artifacts/`; both directories are ignored by Git.

## First planned milestone

Audit UNSW-NB15 for schema issues, duplicates, missing values, leakage, and train/test shift before implementing models.
