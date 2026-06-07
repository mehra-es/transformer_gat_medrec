# Data Directory

## Layout

- `raw/` — Place raw EHR exports here (MIMIC-III/MIMIC-IV, etc.).
- `processed/` — Vocabulary files, DDI graph, and serialized patient sequences.

## Synthetic demo

With `--use_synthetic true`, the pipeline generates in-memory synthetic patients for training and evaluation without MIMIC files.

## Real MIMIC preprocessing

See `src/data/preprocessing.py` for modular hooks. TODO steps for production:

1. Extract diagnoses (ICD), medications (NDC/RxNorm), labs from MIMIC tables.
2. Build visit-level multi-hot vectors and drug vocabulary.
3. Build train/val/test patient splits **by patient_id** (no leakage).
4. Export DDI pairs from a drug interaction database aligned to vocabulary indices.
5. Save `processed/` artifacts consumed by `dataset.py`.
