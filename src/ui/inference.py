"""Shared inference service for the web UI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import collate_patient_batch
from src.data.dataset import MedicationRecommendationDataset
from src.explain import explain_patient
from src.metrics.metrics import compute_all_metrics
from src.data.load_data import apply_data_dims_to_config
from src.train import build_dataloaders, build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_device
from src.utils.seed import set_seed


class InferenceService:
    """Loads model once and serves predictions for the dashboard."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        use_synthetic: bool | None = False,
    ) -> None:
        self.root = ROOT
        self.config_path = Path(config_path or self.root / "config.yaml")
        self.config = load_config(self.config_path)
        set_seed(self.config.get("seed", 42))
        self.device = resolve_device(self.config.get("device", "auto"))
        self.use_synthetic = use_synthetic

        ckpt = checkpoint_path or self.config["paths"]["best_checkpoint"]
        self.checkpoint_path = Path(ckpt)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = self.root / self.checkpoint_path

        _, _, self.test_loader, self.edge_index, self.edge_weight, self.adj_upper, data_meta = (
            build_dataloaders(self.config, use_synthetic)
        )
        if data_meta.get("source") == "mimic_demo":
            self.config = apply_data_dims_to_config(self.config, data_meta)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
        self.adj_upper = self.adj_upper.to(self.device)
        self.threshold = self.config["training"].get("threshold", 0.5)
        self.data_cfg = self.config["data"]

        self.model = build_model(self.config).to(self.device)
        self.model_loaded = False
        if self.checkpoint_path.exists():
            load_checkpoint(self.checkpoint_path, self.model, device=self.device)
            self.model_loaded = True
        self.model.eval()

        self.dataset: MedicationRecommendationDataset = self.test_loader.dataset

    @property
    def num_patients(self) -> int:
        return len(self.dataset)

    def _batch_for_patient(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        batch = collate_patient_batch([sample])
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def patient_summary(self, idx: int) -> Dict[str, Any]:
        """Visit timeline and active feature counts for visualization."""
        sample = self.dataset[idx]
        diag = sample["diagnoses"].numpy()
        med = sample["medications"].numpy()
        lab = sample["labs"].numpy()
        T = diag.shape[0]
        visits = []
        for t in range(T):
            visits.append(
                {
                    "visit": t + 1,
                    "num_diagnoses": int(diag[t].sum()),
                    "num_medications": int(med[t].sum()),
                    "lab_mean": float(lab[t].mean()),
                    "lab_std": float(lab[t].std()),
                    "diagnosis_ids": np.where(diag[t] > 0)[0].tolist()[:8],
                    "medication_ids": np.where(med[t] > 0)[0].tolist()[:8],
                }
            )
        target_ids = np.where(sample["target"].numpy() > 0)[0].tolist()
        return {
            "patient_id": sample["patient_id"],
            "num_visits": T,
            "visits": visits,
            "ground_truth_meds": target_ids,
        }

    @torch.no_grad()
    def predict(self, idx: int, top_k: int = 15) -> Dict[str, Any]:
        batch = self._batch_for_patient(idx)
        out = self.model(
            batch["diagnoses"],
            batch["medications"],
            batch["labs"],
            batch["visit_mask"],
            self.edge_index,
            self.edge_weight,
        )
        probs = out["probs"][0].cpu().numpy()
        gate = out["gate"][0].cpu().numpy()
        gate_mean = float(np.mean(gate))
        patient_weight = gate_mean
        drug_weight = 1.0 - gate_mean

        order = np.argsort(-probs)
        top_idx = order[:top_k]
        pred_binary = (probs >= self.threshold).astype(int)
        pred_ids = np.where(pred_binary > 0)[0].tolist()

        ddi_pairs = self._ddi_pairs(pred_ids)
        target = batch["target"][0].cpu().numpy()
        gt_ids = np.where(target > 0)[0].tolist()
        overlap = sorted(set(pred_ids) & set(gt_ids))

        recommendations = [
            {
                "drug_id": int(i),
                "probability": float(probs[i]),
                "selected": bool(probs[i] >= self.threshold),
            }
            for i in top_idx
        ]

        return {
            "patient_id": batch["patient_id"][0],
            "recommendations": recommendations,
            "predicted_at_threshold": pred_ids,
            "ground_truth": gt_ids,
            "overlap": overlap,
            "fusion": {
                "patient_weight": patient_weight,
                "drug_weight": drug_weight,
                "gate_mean": gate_mean,
            },
            "ddi_pairs_in_prediction": ddi_pairs,
            "threshold": self.threshold,
        }

    def _ddi_pairs(self, drug_ids: List[int]) -> List[Dict[str, Any]]:
        adj = self.adj_upper.cpu().numpy()
        pairs = []
        for i in range(len(drug_ids)):
            for j in range(i + 1, len(drug_ids)):
                a, b = drug_ids[i], drug_ids[j]
                lo, hi = (a, b) if a < b else (b, a)
                w = float(adj[lo, hi]) if lo < adj.shape[0] and hi < adj.shape[1] else 0.0
                if w > 0:
                    pairs.append({"drug_a": a, "drug_b": b, "severity": w})
        return pairs

    @torch.no_grad()
    def evaluate_test_set(self) -> Dict[str, float]:
        all_probs, all_targets = [], []
        for batch in self.test_loader:
            out = self.model(
                batch["diagnoses"].to(self.device),
                batch["medications"].to(self.device),
                batch["labs"].to(self.device),
                batch["visit_mask"].to(self.device),
                self.edge_index,
                self.edge_weight,
            )
            all_probs.append(out["probs"])
            all_targets.append(batch["target"].to(self.device))
        probs = torch.cat(all_probs, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return compute_all_metrics(targets, probs, self.threshold, self.adj_upper)

    def explain(self, idx: int) -> Dict[str, Any]:
        if not self.model_loaded:
            return {"error": "Checkpoint not found. Train the model first."}
        return explain_patient(
            self.config,
            str(self.checkpoint_path),
            self.use_synthetic,
            patient_idx=idx,
        )

    def status(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.model_loaded,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "device": self.device,
            "num_test_patients": self.num_patients,
            "use_synthetic": self.use_synthetic,
            "disclaimer": (
                "Clinical decision support only. Physician review required before medication use."
            ),
        }
