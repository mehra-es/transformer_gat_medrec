# Data Directory

## Default: MIMIC-III Demo (PhysioNet)

**Source:** [MIMIC-III Clinical Database Demo v1.4](https://physionet.org/content/mimiciii-demo/1.4/)

Open-access demo (100 patients). Downloaded automatically on first train/preprocess run.

| Path | Contents |
|------|----------|
| `raw/mimiciii-demo/` | Downloaded ZIP + extracted CSVs (not committed to git) |
| `processed/mimic_demo/` | Preprocessed train/val/test splits + DDI graph |

### Manual preprocess

```bash
.venv/bin/python scripts/preprocess_mimic_demo.py --config config.yaml
```

### What is built from MIMIC CSVs

- **Visits** = hospital admissions (`hadm_id`), ordered by time per `subject_id`
- **Diagnoses** = ICD-9 codes from `DIAGNOSES_ICD.csv`
- **Medications** = generic drug names from `PRESCRIPTIONS.csv`
- **Labs** = top lab `itemid` means from `LABEVENTS.csv`
- **Target** = medications on the patient's final admission
- **DDI graph** = train-set co-prescription pairs (demo has no external DDI DB)

### Synthetic fallback

For quick tests without PhysioNet:

```bash
.venv/bin/python src/main.py --mode train --use_synthetic true
```

## Full MIMIC-III / MIMIC-IV

Credentialed access via PhysioNet is required for the full database. Extend `src/data/preprocessing.py` for production pipelines.
