"""Standard GGUF v3 Binary Exporter & Loader for Path A `.sidecar.gguf` (`~10–25 MB`).

Writes and reads spec-compliant GGUF v3 files (`magic=b"GGUF"`, `version=3`, 32-byte
data alignment, GGML reversed `ne` dimension header) with zero external pip packages,
supporting `q8_0`, `q4_0`, `f16`, and `f32` weight storage compatible with `cncm-rs`
(`candle_core::quantized::gguf_file`).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Tuple

import numpy as np
import torch

from cncm.sidecar import CNCMDeltaNetSidecar, SidecarConfig


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGUF metadata value types
GGUF_VAL_UINT32 = 4
GGUF_VAL_INT32 = 5
GGUF_VAL_FLOAT32 = 6
GGUF_VAL_STRING = 8
GGUF_VAL_ARRAY = 9

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8


def _align_up(offset: int, alignment: int = GGUF_ALIGNMENT) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _pack_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _read_string(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def quantize_rows_q8_0(arr_f32: np.ndarray) -> bytes:
    """Quantize 2D `[rows, cols]` (`cols % 32 == 0`) to GGML `Q8_0` (`34 B / 32 weights`)."""
    flat = np.ascontiguousarray(arr_f32, dtype=np.float32).reshape(-1, 32)
    amax = np.max(np.abs(flat), axis=1, keepdims=True)
    d = (amax / 127.0).astype(np.float16)  # [N_blk, 1]
    d_f32 = d.astype(np.float32)
    inv_d = np.divide(1.0, d_f32, out=np.zeros_like(d_f32), where=d_f32 > 0.0)
    qs = np.clip(np.rint(flat * inv_d), -128, 127).astype(np.int8)

    out = np.empty((flat.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.view(np.uint8)
    out[:, 2:] = qs.view(np.uint8)
    return out.tobytes()


def dequantize_rows_q8_0(raw: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    n_elem = int(np.prod(shape))
    n_blk = n_elem // 32
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(n_blk, 34)
    d = buf[:, :2].copy().view(np.float16).astype(np.float32)  # [n_blk, 1]
    qs = buf[:, 2:].view(np.int8).astype(np.float32)           # [n_blk, 32]
    return (d * qs).reshape(shape)


def quantize_rows_q4_0(arr_f32: np.ndarray) -> bytes:
    """Quantize 2D `[rows, cols]` (`cols % 32 == 0`) to GGML `Q4_0` (`18 B / 32 weights`).

    Matches standard GGML `block_q4_0` layout (`cncm-rs` `dequant_q4_0` / `GgmlDType::Q4_0`):
    - 2-byte FP16 scale `d`
    - 16 nibble bytes `qs[0..16]`: low nibble = `q[j] + 8` (`j=0..15`), high nibble = `q[j+16] + 8`.
    """
    flat = np.ascontiguousarray(arr_f32, dtype=np.float32).reshape(-1, 32)
    amax = np.max(np.abs(flat), axis=1, keepdims=True)
    d = np.where(amax == 0.0, np.float16(0.0), (amax / 7.5).astype(np.float16))
    d_f32 = d.astype(np.float32)
    inv_d = np.divide(1.0, d_f32, out=np.zeros_like(d_f32), where=d_f32 > 0.0)

    q_signed = np.clip(np.rint(flat * inv_d), -8, 7).astype(np.int16)
    q_u = (q_signed + 8).astype(np.uint8)  # [N_blk, 32] in [0, 15]
    lo = q_u[:, :16]
    hi = q_u[:, 16:] << 4
    packed = lo | hi  # [N_blk, 16]

    out = np.empty((flat.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d.view(np.uint8)
    out[:, 2:] = packed
    return out.tobytes()


def dequantize_rows_q4_0(raw: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    n_elem = int(np.prod(shape))
    n_blk = n_elem // 32
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(n_blk, 18)
    d = buf[:, :2].copy().view(np.float16).astype(np.float32)  # [n_blk, 1]
    qs = buf[:, 2:]                                            # [n_blk, 16]
    lo = (qs & 0x0F).astype(np.int16) - 8
    hi = ((qs >> 4) & 0x0F).astype(np.int16) - 8
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)    # [n_blk, 32]
    return (d * q).reshape(shape)


def _encode_kv_entry(key: str, val_type: int, val: Any) -> bytes:
    out = bytearray(_pack_string(key))
    out += struct.pack("<I", val_type)
    if val_type == GGUF_VAL_UINT32:
        out += struct.pack("<I", int(val))
    elif val_type == GGUF_VAL_INT32:
        out += struct.pack("<i", int(val))
    elif val_type == GGUF_VAL_FLOAT32:
        out += struct.pack("<f", float(val))
    elif val_type == GGUF_VAL_STRING:
        out += _pack_string(str(val))
    elif val_type == GGUF_VAL_ARRAY:
        elem_type, items = val
        out += struct.pack("<IQ", elem_type, len(items))
        for it in items:
            if elem_type == GGUF_VAL_UINT32:
                out += struct.pack("<I", int(it))
            elif elem_type == GGUF_VAL_INT32:
                out += struct.pack("<i", int(it))
            elif elem_type == GGUF_VAL_FLOAT32:
                out += struct.pack("<f", float(it))
            else:
                raise ValueError(f"Unsupported GGUF array elem_type={elem_type}")
    else:
        raise ValueError(f"Unsupported GGUF val_type={val_type}")
    return bytes(out)


def _read_kv_value(f: BinaryIO, val_type: int) -> Any:
    if val_type == GGUF_VAL_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    if val_type == GGUF_VAL_INT32:
        return struct.unpack("<i", f.read(4))[0]
    if val_type == GGUF_VAL_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if val_type == GGUF_VAL_STRING:
        return _read_string(f)
    if val_type == GGUF_VAL_ARRAY:
        elem_type, n = struct.unpack("<IQ", f.read(12))
        return [_read_kv_value(f, elem_type) for _ in range(n)]
    raise ValueError(f"Unsupported GGUF metadata type {val_type} when reading sidecar")


def export_sidecar_gguf(
    sidecar: CNCMDeltaNetSidecar,
    out_path: str | Path,
    *,
    quant: str = "q8_0",
) -> Dict[str, Any]:
    """Export `CNCMDeltaNetSidecar` into a compact `.sidecar.gguf` (`q4_0`, `q8_0`, `f16`, or `f32`)."""
    q_mode = quant.strip().lower()
    if q_mode not in ("q4_0", "q8_0", "f16", "f32"):
        raise ValueError(f"Unsupported quant='{quant}' (expected q4_0, q8_0, f16, f32)")

    cfg = sidecar.cfg
    kvs: List[Tuple[str, int, Any]] = [
        ("general.architecture", GGUF_VAL_STRING, "cncm_sidecar"),
        ("general.name", GGUF_VAL_STRING, f"{cfg.donor_id}-cncm-sidecar"),
        ("cncm.sidecar.donor_arch", GGUF_VAL_STRING, cfg.donor_arch),
        ("cncm.sidecar.donor_id", GGUF_VAL_STRING, cfg.donor_id),
        ("cncm.sidecar.donor_vocab_size", GGUF_VAL_UINT32, cfg.donor_vocab_size),
        ("cncm.sidecar.donor_hidden_size", GGUF_VAL_UINT32, cfg.donor_hidden_size),
        ("cncm.sidecar.donor_num_layers", GGUF_VAL_UINT32, cfg.donor_num_layers),
        ("cncm.sidecar.tap_layers", GGUF_VAL_ARRAY, (GGUF_VAL_UINT32, list(cfg.tap_layers))),
        ("cncm.sidecar.hidden_size", GGUF_VAL_UINT32, cfg.sidecar_hidden_size),
        ("cncm.sidecar.num_layers", GGUF_VAL_UINT32, cfg.sidecar_layers),
        ("cncm.sidecar.num_heads", GGUF_VAL_UINT32, cfg.sidecar_heads),
        ("cncm.sidecar.intermediate_size", GGUF_VAL_UINT32, cfg.sidecar_intermediate_size),
        ("cncm.sidecar.deltanet_chunk_size", GGUF_VAL_UINT32, cfg.deltanet_chunk_size),
        ("cncm.sidecar.chunk_size", GGUF_VAL_UINT32, cfg.chunk_size),
        ("cncm.sidecar.levels", GGUF_VAL_ARRAY, (GGUF_VAL_UINT32, list(cfg.levels))),
        ("cncm.sidecar.d_vq", GGUF_VAL_UINT32, cfg.d_vq),
        ("cncm.sidecar.fsq_levels", GGUF_VAL_ARRAY, (GGUF_VAL_UINT32, list(cfg.fsq_levels))),
        ("cncm.sidecar.dec_layers", GGUF_VAL_UINT32, cfg.dec_layers),
        ("cncm.sidecar.dec_num_heads", GGUF_VAL_UINT32, cfg.dec_num_heads),
        ("cncm.sidecar.dec_intermediate_size", GGUF_VAL_UINT32, cfg.dec_intermediate_size),
        ("cncm.sidecar.pre_quant_norm_groups", GGUF_VAL_UINT32, cfg.pre_quant_norm_groups),
        ("cncm.sidecar.phi_ratio", GGUF_VAL_FLOAT32, cfg.phi_ratio),
        ("cncm.sidecar.matryoshka_cutoffs", GGUF_VAL_ARRAY, (GGUF_VAL_UINT32, list(cfg.matryoshka_cutoffs))),
        ("cncm.sidecar.concept_head_hidden", GGUF_VAL_UINT32, cfg.concept_head_hidden),
        ("cncm.sidecar.router_hidden_size", GGUF_VAL_UINT32, cfg.router_hidden_size),
        ("cncm.sidecar.rms_norm_eps", GGUF_VAL_FLOAT32, cfg.rms_norm_eps),
        ("cncm.sidecar.rope_theta", GGUF_VAL_FLOAT32, cfg.rope_theta),
        ("cncm.sidecar.max_position_embeddings", GGUF_VAL_UINT32, cfg.max_position_embeddings),
        ("cncm.sidecar.pad_token_id", GGUF_VAL_UINT32, max(0, cfg.pad_token_id)),
        ("cncm.sidecar.bos_token_id", GGUF_VAL_UINT32, max(0, cfg.bos_token_id)),
        ("cncm.sidecar.eos_token_id", GGUF_VAL_UINT32, max(0, cfg.eos_token_id)),
    ]

    sd = sidecar.state_dict()
    tensor_records: List[Tuple[str, Tuple[int, ...], int, bytes]] = []
    skip_suffixes = ("radix_basis", "fsq_levels_tensor", "center", "half_span", "max_radius", "max_digit")

    for name, tensor in sd.items():
        if any(name.endswith(sfx) for sfx in skip_suffixes):
            continue
        arr = tensor.detach().cpu().float().numpy()
        shape = tuple(int(d) for d in arr.shape)
        gguf_name = f"cncm.sidecar.{name}"

        # Keep norms, biases, conv kernels, svd_proj, and small matrices (<1024 or cols % 32 != 0) in exact F32/F16
        is_quant_candidate = (
            arr.ndim == 2
            and shape[1] % 32 == 0
            and arr.size >= 1024
            and "svd_proj" not in name
        )
        if is_quant_candidate and q_mode == "q4_0":
            ggml_type = GGML_TYPE_Q4_0
            raw_bytes = quantize_rows_q4_0(arr)
        elif is_quant_candidate and q_mode == "q8_0":
            ggml_type = GGML_TYPE_Q8_0
            raw_bytes = quantize_rows_q8_0(arr)
        elif is_quant_candidate and q_mode == "f16":
            ggml_type = GGML_TYPE_F16
            raw_bytes = np.ascontiguousarray(arr, dtype=np.float16).tobytes()
        else:
            ggml_type = GGML_TYPE_F32
            raw_bytes = np.ascontiguousarray(arr, dtype=np.float32).tobytes()

        tensor_records.append((gguf_name, shape, ggml_type, raw_bytes))

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<IQQ", GGUF_VERSION, len(tensor_records), len(kvs))
    for k, vt, vv in kvs:
        header += _encode_kv_entry(k, vt, vv)

    # Compute relative data offsets (each tensor aligned to 32 bytes relative to data section start)
    tensor_infos = bytearray()
    rel_offsets: List[int] = []
    cur_rel = 0
    for gname, shape, ggml_type, raw_bytes in tensor_records:
        cur_rel = _align_up(cur_rel, GGUF_ALIGNMENT)
        rel_offsets.append(cur_rel)
        tensor_infos += _pack_string(gname)
        tensor_infos += struct.pack("<I", len(shape))
        # GGML stores dimensions in reversed order (innermost contiguous axis first!)
        for dim_val in reversed(shape):
            tensor_infos += struct.pack("<Q", dim_val)
        tensor_infos += struct.pack("<IQ", ggml_type, cur_rel)
        cur_rel += len(raw_bytes)

    pre_data_len = len(header) + len(tensor_infos)
    pad_header = _align_up(pre_data_len, GGUF_ALIGNMENT) - pre_data_len

    with open(out_file, "wb") as f:
        f.write(header)
        f.write(tensor_infos)
        if pad_header > 0:
            f.write(b"\x00" * pad_header)
        written_rel = 0
        for (_, _, _, raw_bytes), target_off in zip(tensor_records, rel_offsets):
            if target_off > written_rel:
                f.write(b"\x00" * (target_off - written_rel))
                written_rel = target_off
            f.write(raw_bytes)
            written_rel += len(raw_bytes)

    size_mb = out_file.stat().st_size / (1024.0 * 1024.0)
    return {
        "path": str(out_file),
        "size_mb": round(size_mb, 3),
        "num_tensors": len(tensor_records),
        "quant": q_mode,
    }


def load_sidecar_gguf(
    gguf_path: str | Path,
    device: str | torch.device = "cpu",
) -> CNCMDeltaNetSidecar:
    """Load a `.sidecar.gguf` file back into a ready-to-run `CNCMDeltaNetSidecar`."""
    path = Path(gguf_path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic {magic!r} in {path}")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")

        meta: Dict[str, Any] = {}
        for _ in range(n_kv):
            k = _read_string(f)
            (vt,) = struct.unpack("<I", f.read(4))
            meta[k] = _read_kv_value(f, vt)

        t_specs: List[Tuple[str, Tuple[int, ...], int, int]] = []
        for _ in range(n_tensors):
            gname = _read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            ggml_dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            py_shape = tuple(int(d) for d in reversed(ggml_dims))
            ggml_type, rel_off = struct.unpack("<IQ", f.read(12))
            t_specs.append((gname, py_shape, ggml_type, rel_off))

        data_start = _align_up(f.tell(), GGUF_ALIGNMENT)

        cfg = SidecarConfig(
            donor_arch=str(meta["cncm.sidecar.donor_arch"]),
            donor_id=str(meta["cncm.sidecar.donor_id"]),
            donor_vocab_size=int(meta["cncm.sidecar.donor_vocab_size"]),
            donor_hidden_size=int(meta["cncm.sidecar.donor_hidden_size"]),
            donor_num_layers=int(meta["cncm.sidecar.donor_num_layers"]),
            tap_layers=tuple(int(x) for x in meta["cncm.sidecar.tap_layers"]),  # type: ignore[arg-type]
            sidecar_hidden_size=int(meta["cncm.sidecar.hidden_size"]),
            sidecar_layers=int(meta["cncm.sidecar.num_layers"]),
            sidecar_heads=int(meta["cncm.sidecar.num_heads"]),
            sidecar_intermediate_size=int(meta["cncm.sidecar.intermediate_size"]),
            deltanet_chunk_size=int(meta["cncm.sidecar.deltanet_chunk_size"]),
            chunk_size=int(meta["cncm.sidecar.chunk_size"]),
            levels=tuple(int(x) for x in meta["cncm.sidecar.levels"]),
            d_vq=int(meta["cncm.sidecar.d_vq"]),
            fsq_levels=tuple(int(x) for x in meta["cncm.sidecar.fsq_levels"]),
            dec_layers=int(meta["cncm.sidecar.dec_layers"]),
            dec_num_heads=int(meta["cncm.sidecar.dec_num_heads"]),
            dec_intermediate_size=int(meta["cncm.sidecar.dec_intermediate_size"]),
            pre_quant_norm_groups=int(meta["cncm.sidecar.pre_quant_norm_groups"]),
            phi_ratio=float(meta["cncm.sidecar.phi_ratio"]),
            matryoshka_cutoffs=tuple(int(x) for x in meta["cncm.sidecar.matryoshka_cutoffs"]),
            concept_head_hidden=int(meta["cncm.sidecar.concept_head_hidden"]),
            router_hidden_size=int(meta["cncm.sidecar.router_hidden_size"]),
            rms_norm_eps=float(meta["cncm.sidecar.rms_norm_eps"]),
            rope_theta=float(meta["cncm.sidecar.rope_theta"]),
            max_position_embeddings=int(meta["cncm.sidecar.max_position_embeddings"]),
            pad_token_id=int(meta["cncm.sidecar.pad_token_id"]),
            bos_token_id=int(meta["cncm.sidecar.bos_token_id"]),
            eos_token_id=int(meta["cncm.sidecar.eos_token_id"]),
        )
        sidecar = CNCMDeltaNetSidecar(cfg)
        sd_ref = sidecar.state_dict()

        for gname, py_shape, ggml_type, rel_off in t_specs:
            py_name = gname[len("cncm.sidecar.") :] if gname.startswith("cncm.sidecar.") else gname
            if py_name not in sd_ref:
                continue
            f.seek(data_start + rel_off)
            n_elem = int(np.prod(py_shape))
            if ggml_type == GGML_TYPE_F32:
                arr = np.frombuffer(f.read(n_elem * 4), dtype=np.float32).reshape(py_shape).copy()
            elif ggml_type == GGML_TYPE_F16:
                arr = np.frombuffer(f.read(n_elem * 2), dtype=np.float16).astype(np.float32).reshape(py_shape)
            elif ggml_type == GGML_TYPE_Q8_0:
                arr = dequantize_rows_q8_0(f.read((n_elem // 32) * 34), py_shape)
            elif ggml_type == GGML_TYPE_Q4_0:
                arr = dequantize_rows_q4_0(f.read((n_elem // 32) * 18), py_shape)
            else:
                raise ValueError(f"Unsupported tensor ggml_type={ggml_type} for {gname}")
            sd_ref[py_name].copy_(torch.from_numpy(arr))

    return sidecar.to(device).eval()
