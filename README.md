# Transformer-GAT Medication Recommendation

Research-grade **clinical decision support** framework combining a Transformer temporal encoder, DDI-aware Graph Attention Network (GAT), learnable gated fusion, and DeepSHAP explainability for longitudinal EHR medication recommendation.

> **Not autonomous prescribing.** All outputs require physician review before any medication use.

## Architecture

1. Clinical embedding (diagnosis / medication / lab)
2. Sinusoidal positional encoding
3. Transformer temporal encoder (last valid visit pooling)
4. DDI-aware GAT on drug graph
5. Learnable gated fusion
6. Multi-label recommendation head
7. Loss: `L_total = λ1·L_rec + λ2·L_DDI`
8. DeepSHAP explanations

## Quick start (one script)

### Linux / macOS

From `transformer_gat_medrec/`:

```bash
./run.sh
```

### Windows (PowerShell or CMD)

```powershell
cd transformer_gat_medrec
.\run.ps1
```

Or double-click / run from **CMD**:

```bat
run.bat
run.bat ui
run.bat stop
```

This runs **setup → train → evaluate → tests → web dashboard** (opens at http://127.0.0.1:8080).

| Task | Linux / macOS | Windows (PowerShell) |
|------|---------------|----------------------|
| Full pipeline | `./run.sh` | `.\run.ps1` or `run.bat` |
| Setup only | `./run.sh setup` | `.\run.ps1 setup` |
| Train | `./run.sh train` | `.\run.ps1 train` |
| Evaluate | `./run.sh eval` | `.\run.ps1 eval` |
| Tests | `./run.sh test` | `.\run.ps1 test` |
| Dashboard | `./run.sh ui` | `.\run.ps1 ui` |
| Stop UI | `./run.sh stop` | `.\run.ps1 stop` |
| Skip train if checkpoint exists | `./run.sh all --skip-train` | `.\run.ps1 all -SkipTrain` |
| Custom port | `./run.sh ui --port 8081` | `.\run.ps1 ui -Port 8081` |
| SHAP demo | `./run.sh explain` | `.\run.ps1 explain` |

If PowerShell blocks scripts, use `run.bat` or run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Setup (manual)

**Linux / macOS:**

```bash
cd transformer_gat_medrec
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows:**

```powershell
cd transformer_gat_medrec
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train (synthetic demo)

From the project root `transformer_gat_medrec/` (after activating the venv):

```bash
source .venv/bin/activate
python src/main.py --config config.yaml --mode train --use_synthetic true
```

Without activating the venv:

```bash
.venv/bin/python src/main.py --config config.yaml --mode train --use_synthetic true
```

Best checkpoint: `checkpoints/best.pt`

## Evaluate

```bash
.venv/bin/python src/evaluate.py --checkpoint checkpoints/best.pt --use_synthetic true
```

Or via main:

```bash
.venv/bin/python src/main.py --config config.yaml --mode eval --use_synthetic true --checkpoint checkpoints/best.pt
```

## Explain (SHAP)

```bash
.venv/bin/python src/explain.py --checkpoint checkpoints/best.pt --use_synthetic true --patient_idx 0
```

## Tests

```bash
.venv/bin/pytest tests/ -v
```

## Web dashboard (UI)

Interactive dashboard for exploring patients, recommendations, fusion weights, DDI safety, SHAP explanations, and test metrics.

```bash
# From transformer_gat_medrec/ (train first so checkpoints/best.pt exists)
.venv/bin/pip install fastapi uvicorn
.venv/bin/python ui/server.py
```

Open **http://127.0.0.1:8080** in your browser.

| View | What it shows |
|------|----------------|
| **Pipeline** | Run all `run.sh` steps (setup, train, eval, test, SHAP) with live logs |
| Overview | Architecture and data flow |
| Patient timeline | Per-visit diagnoses, meds, labs |
| Recommendations | Top drug probabilities vs. ground truth |
| Fusion & safety | Gate balance (Transformer vs. GAT) and DDI pairs |
| Explainability | DeepSHAP feature attributions |
| Evaluation | Test-set Jaccard, F1, PRAUC, DDI rate |

## Ablation

```bash
.venv/bin/python src/ablation/run_ablation.py --config config.yaml --use_synthetic true --variant full
```

Variants: `full`, `no_transformer`, `no_gat`, `no_gated_fusion`, `no_ddi_loss`, or `all`.

## Real MIMIC data

1. Place raw exports in `data/raw/`.
2. Implement TODOs in `src/data/preprocessing.py`.
3. Save processed splits to `data/processed/`.
4. Train with `--use_synthetic false` once loading is implemented.

Patient-level train/val/test splits prevent leakage.

## Metrics

- Jaccard (samples)
- Micro / macro F1
- PRAUC (macro over labels)
- DDI rate on predicted drug pairs

## License / disclaimer

For research and decision support prototyping only. Validate on institutional review and clinical workflows before any deployment.
