import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Modulator(nn.Module):
    def __init__(
        self, hidden_size: int, dropout: float = 0.1, modulation_scale: float = 0.5
    ):
        super().__init__()
        self.modulation_scale = torch.tensor(modulation_scale)
        self.hidden_size = hidden_size

        self.param_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
        )
        nn.init.zeros_(self.param_predictor[-1].weight)
        nn.init.zeros_(self.param_predictor[-1].bias)

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat: torch.Tensor,
        cond_seq: torch.Tensor,
        feat_mask: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, hidden = feat.shape

        q = self.w_q(feat)
        k = self.w_k(cond_seq)
        v = self.w_v(cond_seq)

        similarity = torch.matmul(q, k.transpose(-2, -1)) / (hidden**0.5)
        similarity = similarity.masked_fill(~cond_mask.bool().unsqueeze(1), -1e9)
        weights = torch.softmax(similarity, dim=-1)

        cond_aligned = torch.matmul(weights, v).float()

        combined = torch.cat([feat, cond_aligned], dim=-1)
        scale_pred, bias_pred = self.param_predictor(combined).chunk(2, dim=-1)

        gate = self.gate_net(combined)
        scale = self.modulation_scale * torch.tanh(scale_pred) * gate
        bias = self.modulation_scale * torch.tanh(bias_pred) * gate

        delta = feat * scale + bias

        feat_mask_expanded = feat_mask.unsqueeze(-1).float()
        delta = delta * feat_mask_expanded

        modulated_feat = feat + delta

        if modulated_feat.dtype != feat.dtype:
            modulated_feat = modulated_feat.to(feat.dtype)

        return modulated_feat


class FEAM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        modulation_scale: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.spatial_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.spatiotemporal_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.expression_embedding = nn.Parameter(torch.zeros(hidden_size))
        for emb in [
            self.spatial_embedding,
            self.spatiotemporal_embedding,
            self.expression_embedding,
        ]:
            nn.init.normal_(emb, mean=0, std=0.02)

        self.spatial_modulator = Modulator(hidden_size, dropout, modulation_scale)
        self.spatiotemporal_modulator = Modulator(
            hidden_size, dropout, modulation_scale
        )

        self.expression_from_spatial_modulator = Modulator(
            hidden_size, dropout, modulation_scale
        )
        self.expression_from_st_modulator = Modulator(
            hidden_size, dropout, modulation_scale
        )

        self.reverse_fusion_weight = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        spatial_feat: torch.Tensor,
        spatiotemporal_feat: torch.Tensor,
        expression_feat: torch.Tensor,
        spatial_mask: torch.Tensor,
        spatiotemporal_mask: torch.Tensor,
        expression_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = spatial_feat.shape[0]
        device = spatial_feat.device

        modulated_spatial = self.spatial_modulator(
            spatial_feat, expression_feat, spatial_mask, expression_mask
        )
        modulated_spatiotemporal = self.spatiotemporal_modulator(
            spatiotemporal_feat, expression_feat, spatiotemporal_mask, expression_mask
        )

        expression_by_spatial = self.expression_from_spatial_modulator(
            expression_feat, spatial_feat, expression_mask, spatial_mask
        )
        expression_by_st = self.expression_from_st_modulator(
            expression_feat, spatiotemporal_feat, expression_mask, spatiotemporal_mask
        )

        w = torch.sigmoid(self.reverse_fusion_weight).to(spatial_feat.dtype)
        fused_expression = w * expression_by_spatial + (1 - w) * expression_by_st
        expression_mask_expanded = expression_mask.unsqueeze(-1).to(fused_expression.dtype)
        fused_expression = fused_expression * expression_mask_expanded + expression_feat * (
            1.0 - expression_mask_expanded
        )

        modulated_spatial += self.spatial_embedding.unsqueeze(0).unsqueeze(0)
        modulated_spatiotemporal += self.spatiotemporal_embedding.unsqueeze(
            0
        ).unsqueeze(0)
        fused_expression += self.expression_embedding.unsqueeze(0).unsqueeze(0)

        spatial_lengths = spatial_mask.sum(dim=1).long()
        spatiotemporal_lengths = spatiotemporal_mask.sum(dim=1).long()
        expression_lengths = expression_mask.sum(dim=1).long()
        total_lengths = spatial_lengths + spatiotemporal_lengths + expression_lengths
        max_total_len = total_lengths.max().item()

        fused_features = torch.zeros(
            batch_size,
            max_total_len,
            self.hidden_size,
            device=device,
            dtype=spatial_feat.dtype,
        )
        fused_mask = torch.zeros(
            batch_size, max_total_len, dtype=torch.bool, device=device
        )

        spatial_end = spatial_lengths
        st_end = spatial_end + spatiotemporal_lengths
        expression_end = st_end + expression_lengths
        time_indices = torch.arange(max_total_len, device=device).unsqueeze(0)

        mask_s = time_indices < spatial_end.unsqueeze(1)
        mask_st = (time_indices >= spatial_end.unsqueeze(1)) & (
            time_indices < st_end.unsqueeze(1)
        )
        mask_e = (time_indices >= st_end.unsqueeze(1)) & (
            time_indices < expression_end.unsqueeze(1)
        )

        if mask_s.any():
            fused_features[mask_s] = modulated_spatial[spatial_mask.bool()]
            fused_mask[mask_s] = True
        if mask_st.any():
            fused_features[mask_st] = modulated_spatiotemporal[
                spatiotemporal_mask.bool()
            ]
            fused_mask[mask_st] = True
        if mask_e.any():
            fused_features[mask_e] = fused_expression[expression_mask.bool()]
            fused_mask[mask_e] = True

        return fused_features, fused_mask
