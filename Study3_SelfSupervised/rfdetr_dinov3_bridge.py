"""Native DINOv3 backbone bridge for RF-DETR.

This module deliberately replaces RF-DETR's complete encoder module.  It does
not translate DINOv3 tensors into a DINOv2 model.  The scratch and own-data SSL
arms therefore use the same DINOv3 architecture and differ only in backbone
initialization.
"""

from __future__ import annotations

import hashlib
import json
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


Initialization = Literal["scratch", "public_ssl", "own_data_ssl", "public_domain_ssl"]
DEFAULT_FEATURE_LAYERS = (2, 5, 8, 11)


@dataclass(frozen=True)
class BridgeProvenance:
    initialization: Initialization
    architecture: str
    feature_layers: tuple[int, ...]
    patch_size: int
    embedding_dimension: int
    checkpoint: str | None
    checkpoint_sha256: str | None
    strict_checkpoint_load: bool
    public_pretrained_weight_source: str | None = None
    public_pretrained_weight_sha256: str | None = None
    external_pretrained_backbone_allowed: bool = False


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_teacher_backbone_state(payload: object) -> dict[str, Tensor]:
    """Extract only EMA-backbone tensors from an official DINOv3 eval export."""
    if not isinstance(payload, dict) or not isinstance(payload.get("teacher"), dict):
        raise KeyError("Expected an official DINOv3 checkpoint containing a 'teacher' mapping")

    state: dict[str, Tensor] = {}
    for raw_key, value in payload["teacher"].items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key).removeprefix("module.")
        if key.startswith("backbone."):
            state[key.removeprefix("backbone.")] = value

    if not state:
        raise RuntimeError("The checkpoint contains no 'teacher.backbone' tensors")
    return state


def _load_dinov3_model(
    dinov3_repo: Path,
    architecture: str,
    *,
    pretrained: bool = False,
    weights: Path | None = None,
) -> nn.Module:
    repo = dinov3_repo.expanduser().resolve()
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"Invalid DINOv3 repository (hubconf.py missing): {repo}")
    kwargs = {}
    if weights is not None:
        kwargs["weights"] = str(weights.expanduser().resolve())
    return torch.hub.load(
        str(repo),
        architecture,
        source="local",
        pretrained=pretrained,
        trust_repo=True,
        **kwargs,
    )


class DinoV3FeatureEncoder(nn.Module):
    """Expose native DINOv3 intermediate feature maps in RF-DETR's format."""

    def __init__(
        self,
        *,
        dinov3_repo: Path,
        initialization: Initialization,
        checkpoint: Path | None = None,
        public_weights: Path | None = None,
        architecture: str = "dinov3_vits16",
        feature_layers: Sequence[int] = DEFAULT_FEATURE_LAYERS,
    ) -> None:
        super().__init__()
        if initialization not in ("scratch", "public_ssl", "own_data_ssl", "public_domain_ssl"):
            raise ValueError(f"Unsupported initialization: {initialization!r}")
        if initialization in ("scratch", "public_ssl") and checkpoint is not None:
            raise ValueError(f"The {initialization} arm forbids an own-data teacher checkpoint")
        if initialization in ("own_data_ssl", "public_domain_ssl") and checkpoint is None:
            raise ValueError(f"The {initialization} arm requires a teacher checkpoint")
        if initialization in ("public_ssl", "public_domain_ssl") and public_weights is None:
            raise ValueError(
                "The public_ssl arm requires the approved official DINOv3 weights file. "
                "Pass public_weights after downloading it from Meta."
            )

        layers = tuple(int(index) for index in feature_layers)
        if not layers or sorted(set(layers)) != list(layers):
            raise ValueError("feature_layers must be unique and strictly increasing")

        if public_weights is not None and not public_weights.expanduser().is_file():
            raise FileNotFoundError(f"Official DINOv3 weights not found: {public_weights}")
        self.backbone = _load_dinov3_model(
            dinov3_repo,
            architecture,
            pretrained=(initialization == "public_ssl"),
            weights=public_weights,
        )
        depth = len(self.backbone.blocks)
        if layers[0] < 0 or layers[-1] >= depth:
            raise ValueError(f"feature_layers {layers} fall outside DINOv3 depth {depth}")

        self.feature_layers = layers
        self.patch_size = int(self.backbone.patch_size)
        self.embedding_dimension = int(self.backbone.embed_dim)
        self._out_feature_channels = [self.embedding_dimension] * len(layers)
        self._export = False

        checkpoint_path: Path | None = None
        checkpoint_hash: str | None = None
        strict_load = False
        public_weight_source: str | None = None
        public_weight_hash: str | None = None
        if initialization in ("public_ssl", "public_domain_ssl"):
            # The official hub loader loads the released LVD-1689M DINOv3-S/16
            # backbone. Record a tensor fingerprint for reproducibility.
            public_weight_path = public_weights.expanduser().resolve()
            public_weight_source = str(public_weight_path)
            public_weight_hash = sha256_file(public_weight_path)
        if checkpoint is not None:
            checkpoint_path = checkpoint.expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"DINOv3 teacher checkpoint not found: {checkpoint_path}")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = extract_teacher_backbone_state(payload)
            result = self.backbone.load_state_dict(state, strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(
                    "Strict DINOv3 load returned incompatible keys: "
                    f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
                )
            checkpoint_hash = sha256_file(checkpoint_path)
            strict_load = True

        self.provenance = BridgeProvenance(
            initialization=initialization,
            architecture=architecture,
            feature_layers=layers,
            patch_size=self.patch_size,
            embedding_dimension=self.embedding_dimension,
            checkpoint=str(checkpoint_path) if checkpoint_path else None,
            checkpoint_sha256=checkpoint_hash,
            strict_checkpoint_load=strict_load,
            public_pretrained_weight_source=public_weight_source,
            public_pretrained_weight_sha256=public_weight_hash,
            external_pretrained_backbone_allowed=(initialization in ("public_ssl", "public_domain_ssl")),
        )

    def forward(self, images: Tensor) -> list[Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB input, received {tuple(images.shape)}")
        height, width = images.shape[-2:]
        pad_height = (-height) % self.patch_size
        pad_width = (-width) % self.patch_size
        if pad_height or pad_width:
            images = F.pad(images, (0, pad_width, 0, pad_height))
        features = self.backbone.get_intermediate_layers(
            images,
            n=self.feature_layers,
            reshape=True,
            norm=True,
        )
        return list(features)

    def export(self) -> None:
        self._export = True

    def provenance_dict(self) -> dict:
        return asdict(self.provenance)


def _dinov3_layer_id(parameter_name: str, number_of_layers: int) -> int:
    if ".patch_embed." in parameter_name or parameter_name.endswith(("cls_token", "storage_tokens")):
        return 0
    marker = ".blocks."
    if marker in parameter_name:
        suffix = parameter_name.split(marker, 1)[1]
        try:
            return int(suffix.split(".", 1)[0]) + 1
        except ValueError:
            pass
    return number_of_layers + 1


def _dinov3_no_weight_decay(parameter_name: str) -> bool:
    return any(
        marker in parameter_name
        for marker in ("bias", "norm", "gamma", "pos_embed", "storage_tokens", "cls_token")
    )


def _get_named_param_lr_pairs_dinov3(self, args, prefix: str = "backbone.0") -> dict:
    """RF-DETR optimizer groups with layer-wise decay for native DINOv3 names."""
    number_of_layers = len(self.encoder.backbone.blocks)
    pairs = {}
    for local_name, parameter in self.named_parameters():
        if not local_name.startswith("encoder.") or not parameter.requires_grad:
            continue
        full_name = f"{prefix}.{local_name}"
        layer_id = _dinov3_layer_id(full_name, number_of_layers)
        decay = float(args.lr_vit_layer_decay) ** (number_of_layers + 1 - layer_id)
        pairs[full_name] = {
            "params": parameter,
            "lr": float(args.lr_encoder) * decay * float(args.lr_component_decay) ** 2,
            "weight_decay": 0.0 if _dinov3_no_weight_decay(full_name) else float(args.weight_decay),
        }
    return pairs


def install_dinov3_encoder(
    rf_model,
    *,
    dinov3_repo: Path,
    initialization: Initialization,
    checkpoint: Path | None = None,
    public_weights: Path | None = None,
    architecture: str = "dinov3_vits16",
    feature_layers: Sequence[int] = DEFAULT_FEATURE_LAYERS,
) -> DinoV3FeatureEncoder:
    """Replace an instantiated RF-DETR encoder with a native DINOv3 encoder.

    The projector and detector remain intact.  For RF-DETR Small, both DINOv2-S
    and DINOv3-S expose 384 channels, so the existing projector is compatible.
    """
    try:
        detector = rf_model.model.model
    except AttributeError as exc:
        raise TypeError("Unsupported RF-DETR object; expected rf_model.model.model.backbone[0]") from exc
    detector_pretrain = getattr(getattr(rf_model, "model_config", None), "pretrain_weights", None)
    return install_dinov3_encoder_on_detector(
        detector,
        dinov3_repo=dinov3_repo,
        initialization=initialization,
        checkpoint=checkpoint,
        public_weights=public_weights,
        architecture=architecture,
        feature_layers=feature_layers,
        detector_pretrain_weights=detector_pretrain,
    )


def install_dinov3_encoder_on_detector(
    detector: nn.Module,
    *,
    dinov3_repo: Path,
    initialization: Initialization,
    checkpoint: Path | None = None,
    public_weights: Path | None = None,
    architecture: str = "dinov3_vits16",
    feature_layers: Sequence[int] = DEFAULT_FEATURE_LAYERS,
    detector_pretrain_weights: object = None,
) -> DinoV3FeatureEncoder:
    """Install DINOv3 into an already-built raw RF-DETR detector module."""
    try:
        rf_backbone = detector.backbone[0]
    except (AttributeError, IndexError, TypeError) as exc:
        raise TypeError("Unsupported RF-DETR detector; expected detector.backbone[0]") from exc
    detector_pretrain = detector_pretrain_weights
    if detector_pretrain is not None:
        raise RuntimeError(
            "All comparison arms require pretrain_weights=None; "
            f"RF-DETR reports {detector_pretrain!r}"
        )

    encoder = DinoV3FeatureEncoder(
        dinov3_repo=dinov3_repo,
        initialization=initialization,
        checkpoint=checkpoint,
        public_weights=public_weights,
        architecture=architecture,
        feature_layers=feature_layers,
    )
    expected_channels = getattr(rf_backbone.encoder, "_out_feature_channels", None)
    if expected_channels is None:
        raise RuntimeError("Existing RF-DETR encoder does not declare _out_feature_channels")
    if list(expected_channels) != encoder._out_feature_channels:
        raise RuntimeError(
            "RF-DETR projector channel mismatch: "
            f"projector expects {list(expected_channels)}, DINOv3 provides {encoder._out_feature_channels}. "
            "Use RF-DETR Small with dinov3_vits16 or rebuild the projector explicitly."
        )

    reference_parameter = next(rf_backbone.projector.parameters(), None)
    if reference_parameter is not None:
        encoder.to(device=reference_parameter.device)
    rf_backbone.encoder = encoder
    rf_backbone.get_named_param_lr_pairs = types.MethodType(
        _get_named_param_lr_pairs_dinov3,
        rf_backbone,
    )
    rf_backbone.dinov3_bridge_provenance = encoder.provenance_dict()
    return encoder


@contextmanager
def patch_rfdetr_training_rebuild(
    *,
    dinov3_repo: Path,
    initialization: Initialization,
    checkpoint: Path | None = None,
    public_weights: Path | None = None,
    architecture: str = "dinov3_vits16",
    feature_layers: Sequence[int] = DEFAULT_FEATURE_LAYERS,
):
    """Inject DINOv3 when RF-DETR 1.9 rebuilds its Lightning training model.

    RF-DETR 1.9 deliberately creates a fresh detector inside ``train()``.
    Replacing only ``rf_model.model.model.backbone`` before calling ``train``
    would therefore be silently discarded. This scoped patch installs the
    requested DINOv3 encoder immediately after that fresh detector is built and
    before Lightning constructs optimizer parameter groups.
    """
    from rfdetr.training import RFDETRModelModule

    original_init = RFDETRModelModule.__init__

    def patched_init(module_self, model_config, train_config):
        original_init(module_self, model_config, train_config)
        encoder = install_dinov3_encoder_on_detector(
            module_self.model,
            dinov3_repo=dinov3_repo,
            initialization=initialization,
            checkpoint=checkpoint,
            public_weights=public_weights,
            architecture=architecture,
            feature_layers=feature_layers,
            detector_pretrain_weights=model_config.pretrain_weights,
        )
        module_self.dinov3_bridge_provenance = encoder.provenance_dict()

    RFDETRModelModule.__init__ = patched_init
    try:
        yield
    finally:
        RFDETRModelModule.__init__ = original_init


def train_rfdetr_with_dinov3(
    rf_model,
    *,
    dinov3_repo: Path,
    initialization: Initialization,
    checkpoint: Path | None = None,
    public_weights: Path | None = None,
    architecture: str = "dinov3_vits16",
    feature_layers: Sequence[int] = DEFAULT_FEATURE_LAYERS,
    **train_kwargs,
) -> DinoV3FeatureEncoder:
    """Run RF-DETR training while preserving native DINOv3 initialization."""
    configured_pretrain = getattr(getattr(rf_model, "model_config", None), "pretrain_weights", None)
    if configured_pretrain is not None:
        raise RuntimeError(
            "RF-DETR must be constructed with pretrain_weights=None for the main comparison arms"
        )
    with patch_rfdetr_training_rebuild(
        dinov3_repo=dinov3_repo,
        initialization=initialization,
        checkpoint=checkpoint,
        public_weights=public_weights,
        architecture=architecture,
        feature_layers=feature_layers,
    ):
        rf_model.train(**train_kwargs)

    trained_encoder = rf_model.model.model.backbone[0].encoder
    if not isinstance(trained_encoder, DinoV3FeatureEncoder):
        raise RuntimeError("RF-DETR training returned without the native DINOv3 encoder")
    expected = initialization
    observed = trained_encoder.provenance.initialization
    if observed != expected:
        raise RuntimeError(f"DINOv3 initialization changed during training: {observed!r} != {expected!r}")
    return trained_encoder


def write_bridge_provenance(encoder: DinoV3FeatureEncoder, output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(encoder.provenance_dict(), indent=2), encoding="utf-8")
