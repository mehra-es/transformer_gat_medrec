# Transformer-GAT Medication Recommendation

Research-grade **clinical decision support** framework for safe and explainable medication recommendation from longitudinal EHR data, aligned with the architecture specified in *A Transformer- and Graph-Based Framework for Safe and Explainable Medication Recommendation*.

> **Not autonomous prescribing.** All outputs require physician review before any medication use.

## Architecture (seven modules)

| Module | Component | Status in this repo |
|--------|-----------|---------------------|
| 1 | Clinical embedding + sinusoidal positional encoding | Implemented |
| 2 | Transformer temporal encoder (L=4, h=8) | Implemented |
| 3 | Learnable gated fusion | Implemented |
| 4 | Dual-graph drug module (DDI-GAT + molecular substructure GNN) | Implemented |
| 5 | Multi-objective loss (L_rec + L_DDI + L_outcome + L_xai) | L_rec + L_DDI active; outcome/XAI stubs |
| 6 | Seven-method explainability | **DeepSHAP only** (IG, LRP, GNNExplainer deferred) |
| 7 | TCN outcome monitoring | Architectural spec (`src/models/outcome_tcn.py`) |

### Loss function

```
L_total = λ₁·L_rec + λ₂·L_DDI + λ₃·L_outcome + λ₄·L_xai
```

Default weights: λ₁=1.0, λ₂=0.5, λ₃=0.3, λ₄=0.1. `L_outcome` and `L_xai` evaluate to zero until Module 7 and Integrated Gradients are wired.

## Quick start (one script)

### Linux / macOS

```bash
./run.sh
```

### Windows

```powershell
.\run.ps1
```

Runs **setup → train → evaluate → tests → web dashboard** at http://127.0.0.1:8080.

| Task | Command |
|------|---------|
| Full pipeline | `./run.sh` |
| Train | `./run.sh train` |
| Evaluate | `./run.sh eval` |
| SHAP explain | `./run.sh explain` |
| Four-seed validation | `.venv/bin/python scripts/run_multi_seed.py` |
| λ_ddi Pareto sweep | `.venv/bin/python scripts/pareto_sweep.py` |
| Ablation study | `.venv/bin/python src/ablation/run_ablation.py --variant all` |

## Hyperparameters (paper Table 7)

| Parameter | Value |
|-----------|-------|
| Learning rate | 1×10⁻⁴ |
| Batch size | 64 |
| d_model | 256 |
| Transformer layers | 4 |
| Attention heads | 8 |
| GAT layers | 3 |
| Dropout | 0.30 |
| Max epochs | 50 |
| λ₁ (L_rec) | 1.0 |
| λ₂ (L_DDI) | 0.5 |
| Seeds | 42, 1, 123, 2024 |

## Train

Default data: [MIMIC-III Demo v1.4](https://physionet.org/content/mimiciii-demo/1.4/) (auto-download).

```bash
source .venv/bin/activate
python src/main.py --config config.yaml --mode train
```

Synthetic fallback:

```bash
python src/main.py --config config.yaml --mode train --use_synthetic true
```

## Evaluate

```bash
python src/evaluate.py --checkpoint checkpoints/best.pt
```

Reports Jaccard, F1-micro/macro, PRAUC, DDI rate, and Safety-Adjusted Effectiveness (SAE = Jaccard × (1 − DDI_rate)).

## Explain (DeepSHAP)

Patient-specific attribution against each patient's top-predicted drugs (visit-mask bug fixed per paper Section 8.5):

```bash
python src/explain.py --checkpoint checkpoints/best.pt --patient_idx 0
```

## Ablation variants

`full`, `no_transformer`, `no_gat`, `no_molecular_gnn`, `no_gated_fusion`, `no_ddi_loss`

## Data

| Mode | Source |
|------|--------|
| **mimic_demo** (default) | PhysioNet open demo |
| **synthetic** | Local generator for quick tests |

Full MIMIC-III GAMENet cohort (5,430 patients, 153 meds): extend `src/data/preprocessing.py` with credentialed access.

## License / disclaimer

For research and decision support prototyping only. Validate on institutional review and clinical workflows before any deployment.
