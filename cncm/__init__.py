"""CNCM: Compact Causal-Hybrid Next-Concept Model."""

from __future__ import annotations

from cncm.config import CNCMConfig, get_preset, list_presets
from cncm.text_tokenizer import CNCMTextTokenizer

__all__ = [
    "CNCMConfig",
    "CNCMConceptTokenizer",
    "CNCMModel",
    "CNCMTextTokenizer",
    "CopyAwareKeywordCrossAttention",
    "MultiscaleRFSQ",
    "get_preset",
    "list_presets",
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
    raise AttributeError(f"module 'cncm' has no attribute {name!r}")
