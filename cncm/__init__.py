"""CNCM: Compact Causal-Hybrid Next-Concept Model."""

from __future__ import annotations

from cncm.config import CNCMConfig, get_preset, list_presets
from cncm.text_tokenizer import CNCMTextTokenizer

__all__ = [
    "CNCMConfig",
    "CNCMConceptTokenizer",
    "CNCMDeltaNetAttention",
    "CNCMDeltaNetBlock",
    "CNCMDeltaNetSidecar",
    "CNCMModel",
    "CNCMTextTokenizer",
    "CopyAwareKeywordCrossAttention",
    "DonorConfig",
    "FrozenDonorRunner",
    "LearnedDualModeRouter",
    "MultiscaleRFSQ",
    "OfflineDonorCache",
    "SVDConceptTokenizer",
    "SidecarConfig",
    "export_sidecar_gguf",
    "get_preset",
    "list_presets",
    "load_sidecar_gguf",
    "load_universal_donor",
]


def __getattr__(name: str):
    if name == "CNCMConceptTokenizer":
        from cncm.tokenizer_model import CNCMConceptTokenizer

        return CNCMConceptTokenizer
    if name == "CopyAwareKeywordCrossAttention":
        from cncm.tokenizer_model import CopyAwareKeywordCrossAttention

        return CopyAwareKeywordCrossAttention
    if name == "MultiscaleRFSQ":
        from cncm.quantizer import MultiscaleRFSQ

        return MultiscaleRFSQ
    if name == "CNCMModel":
        from cncm.model import CNCMModel

        return CNCMModel
    if name in ("CNCMDeltaNetAttention", "CNCMDeltaNetBlock"):
        from cncm import deltanet

        return getattr(deltanet, name)
    if name in ("DonorConfig", "FrozenDonorRunner", "OfflineDonorCache", "load_universal_donor"):
        from cncm import donor

        return getattr(donor, name)
    if name in (
        "CNCMDeltaNetSidecar",
        "LearnedDualModeRouter",
        "SVDConceptTokenizer",
        "SidecarConfig",
    ):
        from cncm import sidecar

        return getattr(sidecar, name)
    if name in ("export_sidecar_gguf", "load_sidecar_gguf"):
        from cncm import sidecar_gguf

        return getattr(sidecar_gguf, name)
    raise AttributeError(f"module 'cncm' has no attribute {name!r}")
