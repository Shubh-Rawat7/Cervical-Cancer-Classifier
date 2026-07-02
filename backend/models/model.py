"""Herlev cervical cell classifier with a Mamba-based multi-scale fusion head."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BACKBONE_ALIASES, CLASS_NAMES, DEFAULT_BACKBONES, DEFAULT_IMAGE_SIZE, NUM_CLASSES

try:
    import timm
except Exception as exc:  # pragma: no cover - optional dependency
    timm = None
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None

try:
    from mamba_ssm import Mamba as NativeMamba
except Exception:  # pragma: no cover - optional dependency
    NativeMamba = None


ACTIVATION_NAMES = {"silu", "gelu", "mish"}


def make_activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    return nn.SiLU()


def resolve_backbone_name(preferred: str) -> str:
    if timm is None:
        raise ImportError("timm is required for the Herlev Mamba classifier") from _TIMM_IMPORT_ERROR

    available = set(timm.list_models())
    candidates = [preferred]
    candidates.extend(BACKBONE_ALIASES.get(preferred, []))
    candidates.extend(DEFAULT_BACKBONES)

    for candidate in candidates:
        if candidate in available:
            return candidate
        matches = timm.list_models(f"*{candidate}*")
        if matches:
            return matches[0]

    raise ValueError(f"Could not resolve backbone '{preferred}' in timm")


class LiteMambaBlock(nn.Module):
    """Fallback mixer used when mamba-ssm is unavailable."""

    def __init__(self, dim: int, dropout: float = 0.1, activation: str = "silu"):
        super().__init__()
        hidden = dim * 2
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, hidden * 2)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.out_proj = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)
        value, gate = x.chunk(2, dim=-1)
        value = self.activation(value) * torch.sigmoid(gate)
        value = value.transpose(1, 2)
        value = self.depthwise(value)
        value = value.transpose(1, 2)
        value = self.out_proj(value)
        return residual + self.dropout(value)


class MambaMixerBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, activation: str = "silu"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        if NativeMamba is not None:
            self.core = NativeMamba(d_model=dim, d_state=max(16, dim // 8), d_conv=4, expand=2)
        else:
            self.core = LiteMambaBlock(dim=dim, dropout=dropout, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.core(x)
        return residual + self.dropout(x)


class MultiScaleBackbone(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True):
        super().__init__()
        self.backbone_name = resolve_backbone_name(backbone)
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            features_only=True,
        )
        self.channels = list(self.backbone.feature_info.channels())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feature_maps = self.backbone(x)
        if isinstance(feature_maps, (list, tuple)):
            return list(feature_maps)
        return [feature_maps]


class HerlevMambaClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "vim_base_patch16_224",
        num_classes: int = NUM_CLASSES,
        image_size: int = DEFAULT_IMAGE_SIZE,
        embed_dim: int = 256,
        mamba_layers: int = 2,
        attn_heads: int = 4,
        dropout: float = 0.2,
        activation: str = "silu",
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.mamba_layers = mamba_layers
        self.attn_heads = attn_heads
        self.dropout_rate = dropout
        self.activation_name = activation

        self.encoder = MultiScaleBackbone(backbone=backbone, pretrained=pretrained)
        self.feature_channels = list(self.encoder.channels)
        if not self.feature_channels:
            raise RuntimeError("The selected backbone did not expose feature maps")

        self.grid_sizes = self._build_grid_sizes(len(self.feature_channels))
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, embed_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    make_activation(activation),
                )
                for channels in self.feature_channels
            ]
        )
        self.scale_refiners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    make_activation(activation),
                )
                for _ in self.feature_channels
            ]
        )

        total_tokens = sum(grid * grid for grid in self.grid_sizes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens + 1, embed_dim))
        self.token_dropout = nn.Dropout(dropout)
        self.fusion_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=attn_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mamba_blocks = nn.ModuleList([MambaMixerBlock(embed_dim, dropout=dropout, activation=activation) for _ in range(mamba_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        hidden_dim = max(embed_dim // 2, 128)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._cached_feature_maps: List[torch.Tensor] | None = None
        self._cached_attention: torch.Tensor | None = None
        self._init_parameters()

    def _build_grid_sizes(self, num_scales: int) -> List[int]:
        pattern = [2, 2, 1, 1, 1]
        if num_scales <= len(pattern):
            return pattern[:num_scales]
        return pattern + [1] * (num_scales - len(pattern))

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_tokens(self, x: torch.Tensor, retain_feature_maps: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feature_maps = self.encoder(x)
        if retain_feature_maps:
            for feature_map in feature_maps:
                feature_map.retain_grad()
        self._cached_feature_maps = feature_maps

        tokens: List[torch.Tensor] = []
        for index, feature_map in enumerate(feature_maps):
            projected = self.projections[index](feature_map)
            pooled = F.adaptive_avg_pool2d(projected, output_size=(self.grid_sizes[index], self.grid_sizes[index]))
            pooled = pooled.flatten(2).transpose(1, 2)
            pooled = self.scale_refiners[index](pooled)
            tokens.append(pooled)

        token_sequence = torch.cat(tokens, dim=1)
        cls_token = self.cls_token.expand(token_sequence.size(0), -1, -1)
        token_sequence = torch.cat([cls_token, token_sequence], dim=1)
        token_sequence = token_sequence + self.pos_embed[:, : token_sequence.size(1)]
        token_sequence = self.token_dropout(token_sequence)
        return token_sequence, feature_maps

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False,
        retain_feature_maps: bool = False,
    ):
        tokens, feature_maps = self.extract_tokens(x, retain_feature_maps=retain_feature_maps)
        attn_out, attn_weights = self.fusion_attention(tokens, tokens, tokens, need_weights=True)
        tokens = self.attn_norm(tokens + attn_out)
        for block in self.mamba_blocks:
            tokens = block(tokens)
        features = self.final_norm(tokens[:, 0])
        logits = self.head(features)
        self._cached_attention = attn_weights
        if return_features or return_attention:
            payload = [logits]
            if return_features:
                payload.append(features)
                payload.append(feature_maps)
            if return_attention:
                payload.append(attn_weights)
            return tuple(payload)
        return logits

    @property
    def cached_feature_maps(self) -> List[torch.Tensor] | None:
        return self._cached_feature_maps

    @property
    def cached_attention(self) -> torch.Tensor | None:
        return self._cached_attention


def _strip_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "ema_model.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        stripped[new_key] = value
    return stripped


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "ema_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return _strip_prefix(checkpoint[key])
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return _strip_prefix(checkpoint)
    raise ValueError("Checkpoint does not contain a compatible state dict")


def load_model(checkpoint_path: str | Path, device: Optional[torch.device | str] = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)

    metadata = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    backbone = metadata.get("backbone", checkpoint.get("backbone", "vim_base_patch16_224"))
    num_classes = int(metadata.get("num_classes", checkpoint.get("num_classes", NUM_CLASSES)))
    image_size = int(metadata.get("image_size", checkpoint.get("image_size", DEFAULT_IMAGE_SIZE)))
    embed_dim = int(metadata.get("embed_dim", checkpoint.get("embed_dim", 256)))
    mamba_layers = int(metadata.get("mamba_layers", checkpoint.get("mamba_layers", 2)))
    attn_heads = int(metadata.get("attn_heads", checkpoint.get("attn_heads", 4)))
    dropout = float(metadata.get("dropout", checkpoint.get("dropout", 0.2)))
    activation = str(metadata.get("activation", checkpoint.get("activation", "silu")))

    model = HerlevMambaClassifier(
        backbone=backbone,
        num_classes=num_classes,
        image_size=image_size,
        embed_dim=embed_dim,
        mamba_layers=mamba_layers,
        attn_heads=attn_heads,
        dropout=dropout,
        activation=activation,
        pretrained=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_class_names(checkpoint_path: str | Path | None = None) -> List[str]:
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "class_names" in checkpoint:
                return list(checkpoint["class_names"])
            config = checkpoint.get("config", {})
            if "class_names" in config:
                return list(config["class_names"])
    return list(CLASS_NAMES)
