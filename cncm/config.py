"""Configuration and hardware-targeted presets for CNCM.

CNCM (Compact Causal-Hybrid Next-Concept Model) unifies:
  1. Multiscale Residual Finite Scalar Quantization (M-RFSQ) Concept Tokenizer
  2. Causal-Prefix Block-AR Backbone with Relaxed Recursive Layer Looping
  3. Factorized Joint Concept Head + Shared Exact-AR Head (Dual-Mode Agentic AI)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class CNCMConfig:
    """Complete architectural configuration for Stage 1 (Tokenizer) and Stage 2 (Model)."""

    # Preset label (informational)
    preset_name: str = "nano"

    # Base text vocabulary & special token IDs (kept in strict sync with CNCMTextTokenizer)
    vocab_size: int = 4096
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    system_start_id: int = 4
    user_start_id: int = 5
    assistant_start_id: int = 6
    turn_end_id: int = 7
    think_start_id: int = 8
    think_end_id: int = 9
    tool_call_start_id: int = 10
    tool_call_end_id: int = 11
    tool_resp_start_id: int = 12
    tool_resp_end_id: int = 13
    tools_start_id: int = 14
    tools_end_id: int = 15

    # PG-FIM (Plan-Guided Fill-in-the-Middle) special token IDs (high tail of vocab)
    fim_prefix_id: int = -1
    fim_suffix_id: int = -1
    fim_middle_id: int = -1
    fim_end_id: int = -1
    kw_start_id: int = -1
    kw_end_id: int = -1
    kw_empty_id: int = -1
    mlen_1_id: int = -1
    mlen_2_id: int = -1
    mlen_3_id: int = -1
    mlen_4plus_id: int = -1

    # PG-FIM training, planning & context budget settings
    fim_rate: float = 0.5
    kw_max: int = 3
    kw_empty_prob: float = 0.15
    kw_corrupt_prob: float = 0.15
    fim_prefix_budget: int = 512
    fim_suffix_budget: int = 256
    use_copy_decoder: bool = False
    copy_decoder_heads: int = 2
    copy_kw_drop_prob: float = 0.30

    # Stage 1: Multiscale RFSQ Concept Tokenizer
    chunk_size: int = 64
    levels: Tuple[int, ...] = (1, 2, 4, 8, 12, 20, 36, 64)
    d_vq: int = 128
    fsq_levels: Tuple[int, ...] = (8, 8, 8, 8)
    enc_layers: int = 0
    dec_layers: int = 1
    dec_num_heads: int = 4
    dec_intermediate_size: int = 384
    pre_quant_norm_groups: int = 8
    fine_drop_prob: float = 0.5
    fine_drop_min_keep: int = 1
    matryoshka_cutoffs: Tuple[int, ...] = ()
    commit_beta: float = 0.25
    phi_ratio: float = 0.5
    tie_tokenizer_embeddings: bool = False

    # Stage 2: Causal-Hybrid Recursive Backbone & Dual Heads
    hidden_size: int = 384
    num_unique_layers: int = 7
    num_loops: int = 3
    loop_adapter_rank: int = 32
    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    intermediate_size: int = 1024
    concept_head_hidden: int = 160
    max_position_embeddings: int = 2048
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    # Optional QAT fake-quantized linear layers
    use_qat_linear: bool = False
    qat_weight_bits: int = 8
    qat_act_bits: int = 8

    # Optional DeepSeek-style Fine-Grained Mixture-of-Experts (MoE) FFN
    # num_experts == 0 preserves standard dense SwiGLU; >0 enables 1 Shared + N Routed Top-K experts
    num_experts: int = 0
    num_experts_per_tok: int = 2
    moe_expert_intermediate_size: int = 0
    moe_shared_intermediate_size: int = 0
    moe_aux_loss_weight: float = 0.01

    # Optional Bounded-Memory Sliding-Window Attention (SWA) + Attention Sinks
    # sliding_window=None keeps standard full causal attention; setting e.g. 2048 caps local KV memory.
    sliding_window: Optional[int] = None
    attention_sink_tokens: int = 4
    global_attn_every_n_layers: int = 4

    # Optional Cross-Layer KV Sharing (CLA, e.g. 2 = 2:1 KV sharing -> 2x smaller KV cache & I/O)
    kv_sharing_group_size: int = 1

    # Optional Concept-Compressed Prefix Cache (compress evicted 64-tok chunks to coarse M-RFSQ level)
    concept_kv_compress: bool = False
    concept_kv_level: int = 2  # level index into `levels` (default l_2=4 -> 16x compression per 64-tok chunk)

    def __post_init__(self) -> None:
        # Normalize list inputs (e.g. loaded from JSON) into tuples
        self.levels = tuple(int(x) for x in self.levels)
        self.fsq_levels = tuple(int(x) for x in self.fsq_levels)
        self.matryoshka_cutoffs = tuple(int(x) for x in self.matryoshka_cutoffs)

        if self.vocab_size < 288:
            raise ValueError(
                f"vocab_size must be >= 288 (16 special + 256 byte + 11 FIM + subwords), got {self.vocab_size}"
            )
        # Auto-assign PG-FIM special token IDs at the top of the vocabulary [vocab_size-11 .. vocab_size-1]
        # so byte tokens (16..271) and existing core subword IDs remain unchanged.
        fim_base = self.vocab_size - 11
        if self.fim_prefix_id < 0:
            self.fim_prefix_id = fim_base + 0
        if self.fim_suffix_id < 0:
            self.fim_suffix_id = fim_base + 1
        if self.fim_middle_id < 0:
            self.fim_middle_id = fim_base + 2
        if self.fim_end_id < 0:
            self.fim_end_id = fim_base + 3
        if self.kw_start_id < 0:
            self.kw_start_id = fim_base + 4
        if self.kw_end_id < 0:
            self.kw_end_id = fim_base + 5
        if self.kw_empty_id < 0:
            self.kw_empty_id = fim_base + 6
        if self.mlen_1_id < 0:
            self.mlen_1_id = fim_base + 7
        if self.mlen_2_id < 0:
            self.mlen_2_id = fim_base + 8
        if self.mlen_3_id < 0:
            self.mlen_3_id = fim_base + 9
        if self.mlen_4plus_id < 0:
            self.mlen_4plus_id = fim_base + 10
        if len(self.levels) < 2:
            raise ValueError(f"levels must have at least 2 scales, got {self.levels}")
        if self.levels[-1] != self.chunk_size:
            raise ValueError(
                f"levels[-1] ({self.levels[-1]}) must equal chunk_size ({self.chunk_size})"
            )
        if any(self.levels[i] >= self.levels[i + 1] for i in range(len(self.levels) - 1)):
            raise ValueError(f"levels must be strictly increasing, got {self.levels}")
        if self.levels[0] < 1 or self.chunk_size % self.levels[0] != 0:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be divisible by levels[0] ({self.levels[0]})"
            )
        if len(self.fsq_levels) < 1 or any(l < 2 for l in self.fsq_levels):
            raise ValueError(f"all fsq_levels must be >= 2, got {self.fsq_levels}")
        if self.d_vq % self.pre_quant_norm_groups != 0:
            raise ValueError(
                f"d_vq ({self.d_vq}) must be divisible by pre_quant_norm_groups ({self.pre_quant_norm_groups})"
            )
        if self.d_vq % self.dec_num_heads != 0:
            raise ValueError(
                f"d_vq ({self.d_vq}) must be divisible by dec_num_heads ({self.dec_num_heads})"
            )
        if (self.d_vq // self.dec_num_heads) % 2 != 0:
            raise ValueError(
                f"decoder head_dim ({self.d_vq // self.dec_num_heads}) must be even for RoPE"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        if (self.hidden_size // self.num_attention_heads) % 2 != 0:
            raise ValueError(
                f"head_dim ({self.hidden_size // self.num_attention_heads}) must be even for RoPE"
            )
        if self.num_unique_layers < 1 or self.num_loops < 1:
            raise ValueError("num_unique_layers and num_loops must be >= 1")
        if self.num_experts < 0:
            raise ValueError(f"num_experts must be >= 0, got {self.num_experts}")
        if self.num_experts > 0:
            if not (1 <= self.num_experts_per_tok <= self.num_experts):
                raise ValueError(
                    f"num_experts_per_tok ({self.num_experts_per_tok}) must be in [1, {self.num_experts}]"
                )
            if self.moe_expert_intermediate_size <= 0:
                self.moe_expert_intermediate_size = max(64, self.intermediate_size // 3)
            if self.moe_shared_intermediate_size == 0:
                self.moe_shared_intermediate_size = self.moe_expert_intermediate_size
        if self.kv_sharing_group_size < 1:
            raise ValueError(f"kv_sharing_group_size must be >= 1, got {self.kv_sharing_group_size}")
        if not (0 <= self.concept_kv_level < len(self.levels)):
            raise ValueError(f"concept_kv_level ({self.concept_kv_level}) out of range [0, {len(self.levels) - 1}]")

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def total_concept_tokens(self) -> int:
        return sum(self.levels)

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)

    @property
    def d_q(self) -> int:
        return len(self.fsq_levels)

    @property
    def codebook_size(self) -> int:
        return math.prod(self.fsq_levels)

    @property
    def effective_layers(self) -> int:
        return self.num_unique_layers * self.num_loops

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def dec_head_dim(self) -> int:
        return self.d_vq // self.dec_num_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def mlen_token_ids(self) -> Tuple[int, int, int, int]:
        return (self.mlen_1_id, self.mlen_2_id, self.mlen_3_id, self.mlen_4plus_id)

    def mlen_id_for_chunks(self, num_chunks: int) -> int:
        if num_chunks <= 1:
            return self.mlen_1_id
        if num_chunks == 2:
            return self.mlen_2_id
        if num_chunks == 3:
            return self.mlen_3_id
        return self.mlen_4plus_id

    def chunks_for_mlen_id(self, token_id: int) -> int:
        if token_id == self.mlen_1_id:
            return 1
        if token_id == self.mlen_2_id:
            return 2
        if token_id == self.mlen_3_id:
            return 3
        if token_id == self.mlen_4plus_id:
            return 4
        return 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["levels"] = list(self.levels)
        d["fsq_levels"] = list(self.fsq_levels)
        d["matryoshka_cutoffs"] = list(self.matryoshka_cutoffs)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CNCMConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


_PRESETS: Dict[str, Dict[str, Any]] = {
    # Fast CPU unit-test preset (~0.4M total params)
    "tiny_test": dict(
        preset_name="tiny_test",
        vocab_size=512,
        chunk_size=16,
        levels=(1, 2, 4, 8, 16),
        d_vq=48,
        fsq_levels=(4, 4, 4, 4),
        dec_layers=1,
        dec_num_heads=4,
        dec_intermediate_size=128,
        pre_quant_norm_groups=4,
        hidden_size=96,
        num_unique_layers=2,
        num_loops=2,
        loop_adapter_rank=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        concept_head_hidden=48,
        max_position_embeddings=512,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        use_qat_linear=False,
    ),
    # Low-end Android edge preset (~14.5M total params, <15 MB INT4, 21 effective layers)
    "nano": dict(
        preset_name="nano",
        vocab_size=4096,
        chunk_size=64,
        levels=(1, 2, 4, 8, 12, 20, 36, 64),
        d_vq=128,
        fsq_levels=(8, 8, 8, 8),
        dec_layers=1,
        dec_num_heads=4,
        dec_intermediate_size=384,
        pre_quant_norm_groups=8,
        hidden_size=384,
        num_unique_layers=7,
        num_loops=3,
        loop_adapter_rank=32,
        num_attention_heads=6,
        num_key_value_heads=2,
        intermediate_size=1024,
        concept_head_hidden=160,
        max_position_embeddings=2048,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        use_qat_linear=False,
    ),
    # Consumer iGPU laptop preset (~40M total params, <30 MB INT4, 24 effective layers)
    "mini": dict(
        preset_name="mini",
        vocab_size=4096,
        chunk_size=64,
        levels=(1, 2, 4, 8, 12, 20, 36, 64),
        d_vq=192,
        fsq_levels=(8, 8, 8, 8),
        dec_layers=1,
        dec_num_heads=4,
        dec_intermediate_size=512,
        pre_quant_norm_groups=8,
        hidden_size=512,
        num_unique_layers=12,
        num_loops=2,
        loop_adapter_rank=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=1408,
        concept_head_hidden=128,
        max_position_embeddings=4096,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        use_qat_linear=False,
    ),
    # Desktop / high-end edge preset (~93.9M total params, 24 effective layers)
    "base": dict(
        preset_name="base",
        vocab_size=6400,
        chunk_size=64,
        levels=(1, 2, 4, 8, 12, 20, 36, 64),
        d_vq=256,
        fsq_levels=(8, 8, 8, 8),
        dec_layers=1,
        dec_num_heads=4,
        dec_intermediate_size=768,
        pre_quant_norm_groups=8,
        hidden_size=768,
        num_unique_layers=12,
        num_loops=2,
        loop_adapter_rank=32,
        num_attention_heads=12,
        num_key_value_heads=3,
        intermediate_size=2304,
        concept_head_hidden=160,
        max_position_embeddings=4096,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        use_qat_linear=False,
    ),
    # DeepSeek-style Fine-Grained MoE base preset (~221M total / ~94M active params, 24 effective layers)
    # 1 always-on Shared Expert (768) + 8 Routed Experts (768, Top-2) => active FFN width 2304 + per-loop routers
    "base_moe": dict(
        preset_name="base_moe",
        vocab_size=6400,
        chunk_size=64,
        levels=(1, 2, 4, 8, 12, 20, 36, 64),
        d_vq=256,
        fsq_levels=(8, 8, 8, 8),
        dec_layers=1,
        dec_num_heads=4,
        dec_intermediate_size=768,
        pre_quant_norm_groups=8,
        hidden_size=768,
        num_unique_layers=12,
        num_loops=2,
        loop_adapter_rank=32,
        num_attention_heads=12,
        num_key_value_heads=3,
        intermediate_size=2304,
        concept_head_hidden=160,
        max_position_embeddings=4096,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        use_qat_linear=False,
        num_experts=8,
        num_experts_per_tok=2,
        moe_expert_intermediate_size=768,
        moe_shared_intermediate_size=768,
        moe_aux_loss_weight=0.01,
    ),
    # Equal-total-memory Dense control for `base_moe` (~221.3M total & active params, 24 effective layers)
    # Matches `base_moe` FFN parameter budget (9 * 768 = 6912 dense SwiGLU width) for controlled Dense-vs-MoE A/B.
    "base_dense220": dict(
        preset_name="base_dense220",
        vocab_size=6400,
        chunk_size=64,
        levels=(1, 2, 4, 8, 12, 20, 36, 64),
        d_vq=256,
        fsq_levels=(8, 8, 8, 8),
        dec_layers=1,
        dec_num_heads=4,
        dec_intermediate_size=768,
        pre_quant_norm_groups=8,
        hidden_size=768,
        num_unique_layers=12,
        num_loops=2,
        loop_adapter_rank=32,
        num_attention_heads=12,
        num_key_value_heads=3,
        intermediate_size=6912,
        concept_head_hidden=160,
        max_position_embeddings=4096,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        use_qat_linear=False,
    ),
}


def get_preset(name: str, **overrides: Any) -> CNCMConfig:
    """Return a validated CNCMConfig preset ('tiny_test', 'nano', 'mini', 'base')."""
    key = name.strip().lower()
    if key not in _PRESETS:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(f"Unknown CNCM preset '{name}'. Available presets: {available}")
    kwargs = dict(_PRESETS[key])
    kwargs.update(overrides)
    return CNCMConfig(**kwargs)


def list_presets() -> Tuple[str, ...]:
    """Return all available preset names."""
    return tuple(_PRESETS.keys())


def plan_concept_compressed_prefix(
    total_prefix_len: int,
    recent_window: int,
    chunk_size: int = 64,
    coarse_level_len: int = 4,
    sink_tokens: int = 4,
) -> Dict[str, Any]:
    """Compute exact token slots for hybrid [sinks + concept-compressed old chunks + recent exact window]."""
    if total_prefix_len <= sink_tokens + recent_window:
        return {
            "total_prefix_len": int(total_prefix_len),
            "sink_tokens": min(int(sink_tokens), int(total_prefix_len)),
            "compressed_chunks": 0,
            "compressed_concept_slots": 0,
            "recent_exact_slots": max(0, int(total_prefix_len) - min(int(sink_tokens), int(total_prefix_len))),
            "total_kv_slots": int(total_prefix_len),
            "compression_ratio": 1.0,
        }
    evictable = int(total_prefix_len) - int(sink_tokens) - int(recent_window)
    n_chunks = evictable // int(chunk_size)
    remainder = evictable % int(chunk_size)
    concept_slots = n_chunks * int(coarse_level_len)
    recent_slots = int(recent_window) + remainder
    total_slots = int(sink_tokens) + concept_slots + recent_slots
    return {
        "total_prefix_len": int(total_prefix_len),
        "sink_tokens": int(sink_tokens),
        "compressed_chunks": int(n_chunks),
        "compressed_concept_slots": int(concept_slots),
        "recent_exact_slots": int(recent_slots),
        "total_kv_slots": int(total_slots),
        "compression_ratio": round(float(total_prefix_len) / float(max(1, total_slots)), 3),
    }


def compute_kv_memory_footprint(
    cfg: CNCMConfig,
    seq_len: int,
    *,
    kv_bytes: float = 1.0,
) -> Dict[str, Any]:
    """Compute exact KV cache footprint (MB) accounting for CLA sharing, SWA+sinks, and Concept Compression.

    kv_bytes: 2.0 for FP16, ~1.05 for INT8 (per-token per-head scale overhead), ~0.56 for INT4.
    """
    n_eff = cfg.effective_layers
    # Number of unique KV-producing layers under Cross-Layer KV Sharing (CLA)
    kv_layers = math.ceil(n_eff / float(cfg.kv_sharing_group_size))
    bytes_per_tok_per_layer = 2.0 * cfg.num_key_value_heads * cfg.head_dim * float(kv_bytes)

    total_bytes = 0.0
    local_slots_used = int(seq_len)
    global_slots_used = int(seq_len)

    if cfg.sliding_window is not None and cfg.sliding_window > 0:
        if cfg.concept_kv_compress:
            plan = plan_concept_compressed_prefix(
                total_prefix_len=seq_len,
                recent_window=cfg.sliding_window,
                chunk_size=cfg.chunk_size,
                coarse_level_len=cfg.levels[cfg.concept_kv_level],
                sink_tokens=cfg.attention_sink_tokens,
            )
            local_slots_used = plan["total_kv_slots"]
            global_slots_used = plan["total_kv_slots"]
        else:
            local_slots_used = min(int(seq_len), int(cfg.attention_sink_tokens) + int(cfg.sliding_window))

    for idx in range(kv_layers):
        is_global = cfg.global_attn_every_n_layers > 0 and ((idx + 1) % cfg.global_attn_every_n_layers == 0)
        slots = global_slots_used if is_global else local_slots_used
        total_bytes += slots * bytes_per_tok_per_layer

    full_baseline_bytes = n_eff * int(seq_len) * bytes_per_tok_per_layer
    return {
        "seq_len": int(seq_len),
        "effective_layers": n_eff,
        "unique_kv_layers": kv_layers,
        "kv_sharing_group_size": cfg.kv_sharing_group_size,
        "sliding_window": cfg.sliding_window,
        "concept_kv_compress": cfg.concept_kv_compress,
        "local_kv_slots": local_slots_used,
        "global_kv_slots": global_slots_used,
        "kv_mb": round(total_bytes / (1024.0 * 1024.0), 3),
        "full_baseline_kv_mb": round(full_baseline_bytes / (1024.0 * 1024.0), 3),
        "reduction_factor": round(full_baseline_bytes / max(total_bytes, 1e-9), 3),
    }
