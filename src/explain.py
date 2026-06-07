"""DeepSHAP explainability for medication recommendations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import shap
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import collate_patient_batch
from src.models.transformer_gat_model import TransformerGATMedRec
from src.data.load_data import apply_data_dims_to_config
from src.train import build_dataloaders, build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_device
from src.utils.seed import set_seed


class ShapWrapperModel(torch.nn.Module):
    """
    Wrapper accepting flat patient tensor for SHAP.

    Input layout: [diag_flat | med_seq_flat | lab_flat] per batch row.
    Simplified: concatenated [B, num_diag*T + num_med*T + num_lab*T] is heavy;
    we use fixed T_max from a single patient padded to max_visits in batch.
    """

    def __init__(
        self,
        base_model: TransformerGATMedRec,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_diag: int,
        num_med: int,
        num_lab: int,
        max_visits: int,
        drug_indices: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.num_diag = num_diag
        self.num_med = num_med
        self.num_lab = num_lab
        self.max_visits = max_visits
        self.drug_indices = drug_indices or [0]

        feat_per_visit = num_diag + num_med + num_lab
        self.feat_dim = feat_per_visit * max_visits

    def _unpack(self, flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = flat.size(0)
        T = self.max_visits
        nd, nm, nl = self.num_diag, self.num_med, self.num_lab
        chunk = nd + nm + nl
        flat = flat.view(B, T, chunk)
        diagnoses = flat[..., :nd]
        medications = flat[..., nd : nd + nm]
        labs = flat[..., nd + nm :]
        visit_mask = (diagnoses.abs().sum(-1) + medications.abs().sum(-1) + labs.abs().sum(-1) > 0).float()
        visit_mask = visit_mask.clamp(min=1.0)  # at least one visit for SHAP stability
        return diagnoses, medications, labs, visit_mask

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        diagnoses, medications, labs, visit_mask = self._unpack(flat)
        out = self.base(diagnoses, medications, labs, visit_mask, self.edge_index, self.edge_weight)
        # Return probabilities for selected drugs (columns for SHAP output)
        return out["probs"][:, self.drug_indices]


def explain_patient(
    config: Dict[str, Any],
    checkpoint_path: str,
    use_synthetic: bool | None = False,
    patient_idx: int = 0,
) -> Dict[str, Any]:
    """
    Run DeepSHAP on one patient and return top features and drugs.

    Clinical decision support only — outputs require physician review.
    """
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "auto"))
    data_cfg = config["data"]
    max_visits = data_cfg.get("max_visits", 32)

    _, _, test_loader, edge_index, edge_weight, _, data_meta = build_dataloaders(config, use_synthetic)
    if data_meta.get("source") == "mimic_demo":
        config = apply_data_dims_to_config(config, data_meta)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model = build_model(config).to(device)
    ckpt = Path(checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    load_checkpoint(ckpt, model, device=device)
    model.eval()

    dataset = test_loader.dataset
    sample = dataset[patient_idx]
    batch = collate_patient_batch([sample])

    T = batch["diagnoses"].shape[1]
    nd, nm, nl = data_cfg["num_diag"], data_cfg["num_med"], data_cfg["num_lab"]
    # Pad to max_visits for wrapper
    pad_T = max_visits
    diag = torch.zeros(1, pad_T, nd)
    med = torch.zeros(1, pad_T, nm)
    lab = torch.zeros(1, pad_T, nl)
    diag[:, :T] = batch["diagnoses"]
    med[:, :T] = batch["medications"]
    lab[:, :T] = batch["labs"]

    with torch.no_grad():
        full_out = model(
            diag.to(device), med.to(device), lab.to(device),
            batch["visit_mask"].to(device).new_ones(1, pad_T),
            edge_index, edge_weight,
        )
    probs = full_out["probs"][0].cpu().numpy()
    top_k = config["explain"].get("top_k_drugs", 10)
    top_drugs = np.argsort(-probs)[:top_k].tolist()
    drug_indices = top_drugs[: min(5, len(top_drugs))]

    wrapper = ShapWrapperModel(
        model, edge_index, edge_weight, nd, nm, nl, pad_T, drug_indices
    ).to(device)

    flat_sample = torch.cat([diag, med, lab], dim=2).reshape(1, -1)
    # Background from a few test patients
    n_bg = min(config["explain"].get("num_background", 32), len(dataset))
    bg_list = []
    for i in range(n_bg):
        b = collate_patient_batch([dataset[i]])
        d = torch.zeros(1, pad_T, nd)
        m = torch.zeros(1, pad_T, nm)
        l = torch.zeros(1, pad_T, nl)
        t = b["diagnoses"].shape[1]
        d[:, :t] = b["diagnoses"]
        m[:, :t] = b["medications"]
        l[:, :t] = b["labs"]
        bg_list.append(torch.cat([d, m, l], dim=2).reshape(1, -1))
    background = torch.cat(bg_list, dim=0).to(device)

    try:
        explainer = shap.DeepExplainer(wrapper, background)
        shap_values = explainer.shap_values(flat_sample.to(device))
    except Exception:
        explainer = shap.GradientExplainer(wrapper, background)
        shap_values = explainer.shap_values(flat_sample.to(device))

    if isinstance(shap_values, list):
        shap_arr = np.stack([np.asarray(sv).reshape(-1) for sv in shap_values], axis=0)
    else:
        shap_arr = np.asarray(shap_values).reshape(len(drug_indices), -1)

    feat_names = []
    for t in range(pad_T):
        feat_names.extend([f"diag_{t}_{i}" for i in range(nd)])
        feat_names.extend([f"med_{t}_{i}" for i in range(nm)])
        feat_names.extend([f"lab_{t}_{i}" for i in range(nl)])

    mean_abs = np.abs(shap_arr).mean(axis=0)
    top_feat_idx = np.argsort(-mean_abs)[: config["explain"].get("top_k_features", 10)]
    top_features = [
        {"name": feat_names[i], "shap_importance": float(mean_abs[i])}
        for i in top_feat_idx
    ]

    return {
        "patient_id": sample["patient_id"],
        "recommended_drugs": top_drugs,
        "probabilities": {int(d): float(probs[d]) for d in top_drugs},
        "top_shap_features": top_features,
        "disclaimer": "Clinical decision support only. Physician review required before medication use.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--use_synthetic", type=str, default="false")
    parser.add_argument("--patient_idx", type=int, default=0)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    use_syn = args.use_synthetic.lower() in ("true", "1", "yes")
    result = explain_patient(config, args.checkpoint, use_syn, args.patient_idx)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
