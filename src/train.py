"""Training loop for Transformer-GAT MedRec."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_data import apply_data_dims_to_config, build_dataloaders as _build_dataloaders
from src.losses.medication_loss import MedicationLoss
from src.metrics.metrics import compute_all_metrics
from src.models.transformer_gat_model import TransformerGATMedRec
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_device
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def build_dataloaders(
    config: Dict[str, Any],
    use_synthetic: bool | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build train/val/test loaders and DDI graph tensors."""
    loaders = _build_dataloaders(config, ROOT, use_synthetic)
    return loaders


def build_model(config: Dict[str, Any], ablation: Optional[Dict[str, Any]] = None) -> TransformerGATMedRec:
    """Create model from config and optional ablation overrides."""
    data_cfg = config["data"]
    m_cfg = config["model"]
    ab = ablation or {}
    return TransformerGATMedRec(
        num_diag=data_cfg["num_diag"],
        num_med=data_cfg["num_med"],
        num_lab=data_cfg["num_lab"],
        d_model=m_cfg["d_model"],
        dropout=m_cfg["dropout"],
        transformer_layers=m_cfg["transformer"]["num_layers"],
        transformer_heads=m_cfg["transformer"]["num_heads"],
        transformer_ff=m_cfg["transformer"]["dim_feedforward"],
        gat_layers=m_cfg["gat"]["num_layers"],
        gat_heads=m_cfg["gat"]["heads"],
        gat_dropout=m_cfg["gat"]["dropout"],
        use_transformer=ab.get("use_transformer", True),
        use_gat=ab.get("use_gat", True),
        use_gated_fusion=ab.get("use_gated_fusion", True),
        max_visits=data_cfg.get("max_visits", 512),
    )


def _run_epoch(
    model: TransformerGATMedRec,
    loader: DataLoader,
    criterion: MedicationLoss,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    adj_upper: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n_batches = 0
    all_probs, all_targets = [], []

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    for batch in loader:
        diagnoses = batch["diagnoses"].to(device)
        medications = batch["medications"].to(device)
        labs = batch["labs"].to(device)
        target = batch["target"].to(device)
        visit_mask = batch["visit_mask"].to(device)

        out = model(diagnoses, medications, labs, visit_mask, edge_index, edge_weight)
        loss, _ = criterion(out["probs"], target, out["logits"])

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        all_probs.append(out["probs"].detach())
        all_targets.append(target.detach())

    avg_loss = total_loss / max(n_batches, 1)
    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_all_metrics(targets, probs, threshold, adj_upper)
    return avg_loss, metrics


def train_model(
    config: Dict[str, Any],
    use_synthetic: bool = True,
    ablation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Full training with early stopping on validation Jaccard."""
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "auto"))
    logger = get_logger("train", config["paths"].get("log_dir"))

    train_loader, val_loader, test_loader, edge_index, edge_weight, adj_upper, data_meta = build_dataloaders(
        config, use_synthetic
    )
    if data_meta.get("source") == "mimic_demo":
        config = apply_data_dims_to_config(config, data_meta)
    adj_upper = adj_upper.to(device)

    ab = ablation or {}
    loss_cfg = config["loss"].copy()
    if not ab.get("use_ddi_loss", True):
        loss_cfg["lambda_ddi"] = 0.0

    model = build_model(config, ab).to(device)
    criterion = MedicationLoss(
        lambda_rec=loss_cfg["lambda_rec"],
        lambda_ddi=loss_cfg["lambda_ddi"],
        use_bce_logits=loss_cfg.get("use_bce_logits", True),
    )
    criterion.set_ddi_adjacency(adj_upper)

    t_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
    )

    num_epochs = t_cfg["num_epochs"]
    warmup = t_cfg.get("warmup_epochs", 3)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(max(warmup, 1))
        progress = (epoch - warmup) / float(max(num_epochs - warmup, 1))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_jaccard = -1.0
    patience = t_cfg.get("early_stopping_patience", 15)
    stale = 0
    ckpt_path = Path(config["paths"]["best_checkpoint"])
    threshold = t_cfg.get("threshold", 0.5)

    for epoch in range(num_epochs):
        train_loss, train_metrics = _run_epoch(
            model, train_loader, criterion, edge_index, edge_weight, device,
            optimizer, adj_upper, threshold,
        )
        val_loss, val_metrics = _run_epoch(
            model, val_loader, criterion, edge_index, edge_weight, device,
            None, adj_upper, threshold,
        )
        scheduler.step()

        logger.info(
            "Epoch %d | train_loss=%.4f val_loss=%.4f | val_jaccard=%.4f f1_micro=%.4f "
            "prauc=%.4f ddi_rate=%.4f",
            epoch + 1,
            train_loss,
            val_loss,
            val_metrics["jaccard"],
            val_metrics["f1_micro"],
            val_metrics["prauc"],
            val_metrics["ddi_rate"],
        )

        if val_metrics["jaccard"] > best_jaccard:
            best_jaccard = val_metrics["jaccard"]
            stale = 0
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                config,
            )
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Best validation Jaccard: %.4f", best_jaccard)
    return {"best_jaccard": best_jaccard, "test_loader": test_loader}
