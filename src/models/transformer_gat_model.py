"""Full Transformer-GAT medication recommendation model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.clinical_embedding import ClinicalEmbedding
from src.models.ddi_gat import DDIAwareGAT, aggregate_patient_drug_embedding
from src.models.gated_fusion import GatedFusion
from src.models.positional_encoding import PositionalEncoding
from src.models.recommendation_head import RecommendationHead
from src.models.transformer_encoder import TransformerTemporalEncoder


class BiLSTMTemporalEncoder(nn.Module):
    """Ablation: replace Transformer with BiLSTM."""

    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            d_model,
            d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, visit_embeddings: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
        lengths = visit_mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            visit_embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        last_idx = lengths - 1
        batch_idx = torch.arange(out.size(0), device=out.device)
        h = out[batch_idx, last_idx, :]
        return self.proj(h)


class TransformerGATMedRec(nn.Module):
    """
    End-to-end Transformer-GAT medication recommendation.

    Ablation flags:
        use_transformer: if False, use BiLSTM
        use_gat: if False, use learnable drug embedding table
        use_gated_fusion: if False, concatenate and project
    """

    def __init__(
        self,
        num_diag: int,
        num_med: int,
        num_lab: int,
        d_model: int = 256,
        dropout: float = 0.3,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ff: int = 512,
        gat_layers: int = 3,
        gat_heads: int = 4,
        gat_dropout: float = 0.2,
        use_transformer: bool = True,
        use_gat: bool = True,
        use_gated_fusion: bool = True,
        max_visits: int = 512,
    ) -> None:
        super().__init__()
        self.num_med = num_med
        self.d_model = d_model
        self.use_gat = use_gat
        self.use_gated_fusion = use_gated_fusion

        self.clinical_embedding = ClinicalEmbedding(
            num_diag, num_med, num_lab, d_model, dropout
        )
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_visits, dropout=dropout)

        if use_transformer:
            self.temporal_encoder: nn.Module = TransformerTemporalEncoder(
                d_model=d_model,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                dim_feedforward=transformer_ff,
                dropout=dropout,
            )
        else:
            self.temporal_encoder = BiLSTMTemporalEncoder(d_model, dropout=dropout)

        self.drug_init = nn.Parameter(torch.randn(num_med, d_model) * 0.02)
        if use_gat:
            self.ddi_gat = DDIAwareGAT(d_model, gat_layers, gat_heads, gat_dropout)
        else:
            self.drug_embedding = nn.Embedding(num_med, d_model)
            nn.init.normal_(self.drug_embedding.weight, std=0.02)

        if use_gated_fusion:
            self.fusion = GatedFusion(d_model)
        else:
            self.fusion_concat = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.head = RecommendationHead(d_model, num_med)

    def encode_drugs(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_gat:
            return self.ddi_gat(self.drug_init, edge_index, edge_weight)
        return self.drug_embedding.weight

    def forward(
        self,
        diagnoses: torch.Tensor,
        medications: torch.Tensor,
        labs: torch.Tensor,
        visit_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        visit_emb = self.clinical_embedding(diagnoses, medications, labs)
        visit_emb = self.pos_encoding(visit_emb)
        h_patient = self.temporal_encoder(visit_emb, visit_mask)

        drug_emb = self.encode_drugs(edge_index, edge_weight)

        lengths = visit_mask.sum(dim=1).long().clamp(min=1)
        last_idx = lengths - 1
        batch_idx = torch.arange(medications.size(0), device=medications.device)
        latest_meds = medications[batch_idx, last_idx, :]
        h_drug = aggregate_patient_drug_embedding(drug_emb, latest_meds)

        if self.use_gated_fusion:
            h_fused, gate = self.fusion(h_patient, h_drug)
        else:
            gate = torch.zeros_like(h_patient)
            h_fused = self.fusion_concat(torch.cat([h_patient, h_drug], dim=-1))

        logits, probs = self.head(h_fused)
        return {
            "probs": probs,
            "logits": logits,
            "gate": gate,
            "h_patient": h_patient,
            "h_drug": h_drug,
            "h_fused": h_fused,
        }
