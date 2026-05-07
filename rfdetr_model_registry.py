from __future__ import annotations

from typing import Any, Dict

CORE_RFDETR_MODEL_NAMES = (
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRLarge",
)

PLUS_RFDETR_MODEL_NAMES = (
    "RFDETRXLarge",
    "RFDETR2XLarge",
)

SUPPORTED_RFDETR_MODEL_NAMES = CORE_RFDETR_MODEL_NAMES + PLUS_RFDETR_MODEL_NAMES
SUPPORTED_RFDETR_MODEL_NAME_SET = set(SUPPORTED_RFDETR_MODEL_NAMES)
PLUS_RFDETR_MODEL_NAME_SET = set(PLUS_RFDETR_MODEL_NAMES)

RFDETR_MODEL_NAME_ALIASES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "xlarge": "RFDETRXLarge",
    "xl": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
    "2xl": "RFDETR2XLarge",
}

RFDETR_DEFAULT_RESOLUTIONS = {
    "RFDETRNano": 384,
    "RFDETRSmall": 512,
    "RFDETRMedium": 576,
    "RFDETRLarge": 704,
    "RFDETRXLarge": 700,
    "RFDETR2XLarge": 880,
}


def supported_rfdetr_model_names(include_auto: bool = False) -> tuple[str, ...]:
    if include_auto:
        return ("auto",) + SUPPORTED_RFDETR_MODEL_NAMES
    return SUPPORTED_RFDETR_MODEL_NAMES


def is_plus_rfdetr_model_name(model_name: str) -> bool:
    return str(model_name).strip() in PLUS_RFDETR_MODEL_NAME_SET


def canonical_rfdetr_model_name(model_name_or_cls: Any, default: str = "RFDETRLarge") -> str:
    if isinstance(model_name_or_cls, str):
        raw = model_name_or_cls.strip()
        if raw in SUPPORTED_RFDETR_MODEL_NAME_SET:
            return raw
        return RFDETR_MODEL_NAME_ALIASES.get(raw.lower(), raw or default)

    name = getattr(model_name_or_cls, "__name__", "")
    if name in SUPPORTED_RFDETR_MODEL_NAME_SET:
        return name
    return default


def default_rfdetr_resolution(model_name_or_cls: Any, fallback: int | None = None) -> int | None:
    model_name = canonical_rfdetr_model_name(model_name_or_cls)
    return RFDETR_DEFAULT_RESOLUTIONS.get(model_name, fallback)


def infer_rfdetr_model_name_from_checkpoint_name(checkpoint_name: str, default: str = "RFDETRLarge") -> str:
    low = str(checkpoint_name).strip().lower()
    if not low:
        return default
    if "2xlarge" in low or "2xl" in low:
        return "RFDETR2XLarge"
    if "xlarge" in low or "xl" in low:
        return "RFDETRXLarge"
    if "small" in low:
        return "RFDETRSmall"
    if "medium" in low:
        return "RFDETRMedium"
    if "nano" in low:
        return "RFDETRNano"
    return default


def get_available_rfdetr_model_registry() -> Dict[str, type]:
    registry: Dict[str, type] = {}

    import rfdetr

    for name in CORE_RFDETR_MODEL_NAMES:
        cls = getattr(rfdetr, name, None)
        if cls is not None:
            registry[name] = cls

    try:
        import rfdetr_plus
    except ModuleNotFoundError:
        rfdetr_plus = None

    if rfdetr_plus is not None:
        for name in PLUS_RFDETR_MODEL_NAMES:
            cls = getattr(rfdetr_plus, name, None)
            if cls is not None:
                registry[name] = cls

    return registry


def resolve_rfdetr_model_class(model_name_or_cls: Any) -> type:
    model_name = canonical_rfdetr_model_name(model_name_or_cls)
    registry = get_available_rfdetr_model_registry()
    if model_name in registry:
        return registry[model_name]

    if model_name in PLUS_RFDETR_MODEL_NAME_SET:
        raise ImportError(
            f"{model_name} requires the optional rfdetr_plus package. "
            "Install it with `pip install rfdetr-plus` or `pip install \"rfdetr[plus]\"` "
            "in the target environment."
        )

    raise ImportError(
        f"{model_name} is not available from the installed rfdetr package. "
        f"Supported names are: {', '.join(SUPPORTED_RFDETR_MODEL_NAMES)}."
    )


def instantiate_rfdetr_model(model_name_or_cls: Any, **kwargs: Any) -> Any:
    model_name = canonical_rfdetr_model_name(model_name_or_cls)
    model_cls = resolve_rfdetr_model_class(model_name)
    ctor_kwargs = dict(kwargs)
    if model_name in PLUS_RFDETR_MODEL_NAME_SET:
        ctor_kwargs.setdefault("accept_platform_model_license", True)
    return model_cls(**ctor_kwargs)
