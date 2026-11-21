"""CNCM Edge Quantization (Packed INT4 / INT8), Hardware Profiling, and ONNX Export."""

from __future__ import annotations

__all__ = [
    "Int4PackedLinear",
    "Int8Embedding",
    "Int8Linear",
    "Stage1DecoderONNXWrapper",
    "Stage2ConceptStepONNXWrapper",
    "Stage2ExactStepONNXWrapper",
    "export_cncm_onnx",
    "format_edge_profile_table",
    "measure_model_size",
    "profile_edge_targets",
    "quantize_model_int4",
    "quantize_model_int8",
]

_ONNX_EXPORTS = {
    "Stage1DecoderONNXWrapper",
    "Stage2ConceptStepONNXWrapper",
    "Stage2ExactStepONNXWrapper",
    "export_cncm_onnx",
}


def __getattr__(name: str):
    if name in _ONNX_EXPORTS:
        from cncm.export import export_onnx
        return getattr(export_onnx, name)
    if name in __all__:
        from cncm.export import quantize
        return getattr(quantize, name)
    raise AttributeError(f"module 'cncm.export' has no attribute {name!r}")
