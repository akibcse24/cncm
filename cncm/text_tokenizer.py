"""Zero-dependency lossless byte-fallback + subword text tokenizer for CNCM.

Design goals:
- Fixed special token IDs 0..15 matching CNCMConfig exactly (including agentic
  <think>, <tool_call>, and <tool_response> boundaries).
- Raw byte fallback tokens 16..271 so ANY UTF-8 string round-trips losslessly at
  any vocab_size >= 272 (tiny_test=512, nano/mini=4096, base=6400).
- Built-in deterministic subword vocabulary covering English, JSON/tool-call
  syntax, arithmetic, and common code tokens without external downloads.
- Optional from_hf_json(path) loader for HuggingFace tokenizer.json files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


SPECIAL_TOKENS: Tuple[Tuple[int, str], ...] = (
    (0, "<|pad|>"),
    (1, "<|bos|>"),
    (2, "<|eos|>"),
    (3, "<|unk|>"),
    (4, "<|system|>"),
    (5, "<|user|>"),
    (6, "<|assistant|>"),
    (7, "<|end|>"),
    (8, "<think>"),
    (9, "</think>"),
    (10, "<tool_call>"),
    (11, "</tool_call>"),
    (12, "<tool_response>"),
    (13, "</tool_response>"),
    (14, "<|tools|>"),
    (15, "</|tools|>"),
)

PG_FIM_SPECIAL_STRINGS: Tuple[str, ...] = (
    "<fim_prefix>",
    "<fim_suffix>",
    "<fim_middle>",
    "<fim_end>",
    "<kw>",
    "</kw>",
    "<kw/>",
    "<mlen_1>",
    "<mlen_2>",
    "<mlen_3>",
    "<mlen_4+>",
)

_BYTE_OFFSET = 16
_SUBWORD_OFFSET = _BYTE_OFFSET + 256  # 272
_NUM_FIM_SPECIALS = len(PG_FIM_SPECIAL_STRINGS)  # 11

# Curated high-frequency English, JSON, math, agentic & code subwords.
# Deterministic order ensures stable IDs across runs and presets.
_SEED_SUBWORDS: Tuple[str, ...] = (
    # JSON / tool-call structural tokens
    '{"', '"}', '": "', '": ', '", "', '", ', '":', '",', '[{', '}]',
    'null', 'true', 'false', 'name', 'arguments', 'result', 'error', 'status',
    'calculator', 'kv_memory', 'text_stats', 'clock_utc', 'expression',
    'action', 'key', 'value', 'text', 'query', 'input', 'output', 'tool',
    'get', 'set', 'list', 'delete', 'ok', 'utc', 'time', 'date', 'count',
    # Common whitespace & punctuation pairs
    '  ', '    ', '\n\n', '. ', ', ', ': ', '; ', '? ', '! ', ' - ', ' = ',
    ' + ', ' * ', ' / ', ' -> ', ' == ', ' != ', ' <= ', ' >= ', '()', '[]', '{}',
    # Top English function words & subwords (with leading space & bare forms)
    ' the', ' a', ' an', ' to', ' of', ' in', ' is', ' it', ' for', ' on',
    ' that', ' with', ' as', ' at', ' by', ' from', ' be', ' are', ' was', ' were',
    ' have', ' has', ' had', ' do', ' does', ' did', ' will', ' would', ' can',
    ' could', ' should', ' may', ' might', ' must', ' not', ' no', ' yes',
    ' and', ' or', ' but', ' if', ' then', ' else', ' when', ' while', ' where',
    ' what', ' which', ' who', ' why', ' how', ' this', ' these', ' those',
    ' all', ' any', ' some', ' one', ' two', ' three', ' four', ' five',
    ' six', ' seven', ' eight', ' nine', ' ten', ' zero', ' first', ' second',
    ' next', ' last', ' new', ' old', ' good', ' fast', ' small', ' large',
    ' high', ' low', ' more', ' most', ' less', ' very', ' only', ' also',
    ' just', ' now', ' here', ' there', ' up', ' down', ' out', ' over',
    ' into', ' about', ' after', ' before', ' between', ' under', ' through',
    ' use', ' using', ' used', ' make', ' made', ' compute', ' calculate',
    ' answer', ' step', ' steps', ' plan', ' check', ' verify', ' final',
    ' model', ' token', ' tokens', ' concept', ' concepts', ' level', ' levels',
    ' chunk', ' chunks', ' prefix', ' cache', ' layer', ' layers', ' head',
    ' attention', ' memory', ' edge', ' device', ' mobile', ' laptop', ' agent',
    ' function', ' call', ' return', ' value', ' values', ' number', ' numbers',
    ' string', ' data', ' file', ' code', ' json', ' schema', ' prompt',
    ' user', ' assistant', ' system', ' response', ' question', ' task',
    ' solve', ' result', ' total', ' sum', ' product', ' difference', ' ratio',
    ' average', ' mean', ' max', ' min', ' length', ' size', ' speed',
    'The ', 'A ', 'An ', 'In ', 'On ', 'To ', 'For ', 'With ', 'Let ', 'We ',
    'You ', 'I ', 'It ', 'This ', 'That ', 'Here ', 'Step ', 'Final ', 'Answer',
    'the', 'and', 'for', 'that', 'with', 'this', 'from', 'have', 'not', 'are',
    'but', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out',
    'ing', 'tion', 'ment', 'ness', 'able', 'ible', 'less', 'ful', 'ize', 'ise',
    'ed', 'er', 'es', 'ly', 'al', 'en', 'in', 'on', 'at', 'an', 're', 'de',
    'un', 'pre', 'pro', 'con', 'com', 'sub', 'inter', 'trans', 'super', 'auto',
    # Common bigrams & digits
    'th', 'he', 'er', 're', 'nd', 'st', 'nt', 'es', 'or', 'te', 'of', 'is',
    'it', 'ar', 'to', 'ng', 'se', 'ha', 'as', 'ou', 'io', 'le', 've', 'co',
    'me', 'hi', 'ri', 'ro', 'ic', 'ne', 'ea', 'ra', 'ce', 'li', 'ch', 'll',
    'be', 'ma', 'si', 'om', 'ur', 'ca', 'el', 'ta', 'la', 'ns', 'di', 'fo',
    'ho', 'pe', 'ec', 'pr', 'no', 'ct', 'us', 'ac', 'ot', 'il', 'tr', 'ly',
    '00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
    '10', '11', '12', '13', '14', '15', '16', '20', '24', '25', '32', '42',
    '50', '64', '100', '128', '256', '512', '1024', '2025', '2026', '4096',
    # Code / Python keywords & PG-FIM identifiers
    'def ', 'class ', 'return ', 'import ', 'from ', 'None', 'True', 'False',
    'self.', 'for ', 'while ', 'if ', 'elif ', 'else:', 'try:', 'except ',
    'print(', 'len(', 'range(', 'dict', 'list', 'tuple', 'int', 'float', 'str',
    ' | ', 'return', 'for', 'while', 'if', 'elif', 'else', 'break', 'continue',
    'pass', 'raise', 'yield', 'with', 'as', 'in', 'not', 'and', 'or', 'is',
    'lambda', 'assert', 'async', 'await', 'try', 'except', 'finally', 'idx',
    'val', 'res', 'out', 'acc', 'arr', 'items', 'node', 'left', 'right', 'mid',
    'start', 'end', 'low', 'high', 'curr', 'prev', 'next_val', 'target', 'nums',
)


def _generate_deterministic_subwords(target_count: int) -> List[bytes]:
    """Generate a deterministic, deduplicated list of multi-byte subwords."""
    seen = set()
    tokens: List[bytes] = []

    def _add(s: str) -> None:
        b = s.encode("utf-8")
        if len(b) >= 2 and b not in seen:
            seen.add(b)
            tokens.append(b)

    for s in _SEED_SUBWORDS:
        if len(tokens) >= target_count:
            return tokens
        _add(s)

    # Space-prefixed lowercase letters and frequent ASCII bigrams to fill larger vocabs
    alpha = "etaoinshrdlcumwfgypbvkjxqz"
    digits = "0123456789"

    for ch in alpha:
        if len(tokens) >= target_count:
            return tokens
        _add(" " + ch)
    for ch in alpha.upper():
        if len(tokens) >= target_count:
            return tokens
        _add(" " + ch)
    for d in digits:
        if len(tokens) >= target_count:
            return tokens
        _add(" " + d)
    # 2-char letter bigrams
    for a in alpha:
        for b in alpha:
            if len(tokens) >= target_count:
                return tokens
            _add(a + b)
    # Space + 2-char bigrams
    for a in alpha:
        for b in alpha:
            if len(tokens) >= target_count:
                return tokens
            _add(" " + a + b)
    # 2-char letter + trailing space
    for a in alpha:
        for b in alpha:
            if len(tokens) >= target_count:
                return tokens
            _add(a + b + " ")
    # 3-char trigrams (most frequent letters)
    top = "etaoinshrdlcumwf"
    for a in top:
        for b in top:
            for c in top:
                if len(tokens) >= target_count:
                    return tokens
                _add(a + b + c)

    # Deterministic numbered fallback if vocab_size > ~6500
    idx = 0
    while len(tokens) < target_count:
        _add(f"<|tok_{idx}|>")
        idx += 1

    return tokens


class _ByteTrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: Dict[int, _ByteTrieNode] = {}
        self.token_id: Optional[int] = None


class CNCMTextTokenizer:
    """Lossless byte-fallback + subword tokenizer aligned with CNCMConfig."""

    def __init__(
        self,
        vocab_size: int = 4096,
        custom_subwords: Optional[Sequence[bytes]] = None,
    ) -> None:
        if vocab_size < _SUBWORD_OFFSET + _NUM_FIM_SPECIALS:
            raise ValueError(
                f"vocab_size must be >= {_SUBWORD_OFFSET + _NUM_FIM_SPECIALS}, got {vocab_size}"
            )
        self._vocab_size = int(vocab_size)
        self._fim_base = self._vocab_size - _NUM_FIM_SPECIALS
        self._bpe_merges: Optional[Dict[Tuple[bytes, bytes], int]] = None
        self._bpe_split_digits: bool = False
        self._bpe_word_cache: Dict[bytes, Tuple[int, ...]] = {}
        self._hf_fast_tok: Optional[Any] = None
        self._hf_id_to_cncm_ids: Optional[List[Tuple[int, ...]]] = None

        # Build ID <-> token tables (core 0..15 + PG-FIM reserved buffer slots at tail)
        self._special_id_to_str: Dict[int, str] = {}
        self._special_str_to_id: Dict[str, int] = {}
        for tid, tok in SPECIAL_TOKENS:
            self._special_id_to_str[tid] = tok
            self._special_str_to_id[tok] = tid
        for offset, tok in enumerate(PG_FIM_SPECIAL_STRINGS):
            tid = self._fim_base + offset
            self._special_id_to_str[tid] = tok
            self._special_str_to_id[tok] = tid

        # Byte table for IDs 16..271 and subword table for 272..vocab_size-12
        self._id_to_bytes: Dict[int, bytes] = {}
        self._bytes_to_id: Dict[bytes, int] = {}

        for b in range(256):
            tid = _BYTE_OFFSET + b
            raw = bytes([b])
            self._id_to_bytes[tid] = raw
            self._bytes_to_id[raw] = tid

        n_sub = self._fim_base - _SUBWORD_OFFSET
        if custom_subwords is not None:
            subwords = [w for w in custom_subwords if len(w) >= 2][:n_sub]
            if len(subwords) < n_sub:
                extras = _generate_deterministic_subwords(n_sub * 2)
                existing = set(subwords)
                for w in extras:
                    if w not in existing:
                        subwords.append(w)
                        existing.add(w)
                    if len(subwords) >= n_sub:
                        break
        else:
            subwords = _generate_deterministic_subwords(n_sub)

        for offset, raw in enumerate(subwords):
            tid = _SUBWORD_OFFSET + offset
            self._id_to_bytes[tid] = raw
            self._bytes_to_id[raw] = tid

        # Build longest-match byte trie for O(N) greedy subword encoding
        self._trie_root = _ByteTrieNode()
        for tid, raw in self._id_to_bytes.items():
            node = self._trie_root
            for byte_val in raw:
                nxt = node.children.get(byte_val)
                if nxt is None:
                    nxt = _ByteTrieNode()
                    node.children[byte_val] = nxt
                node = nxt
            node.token_id = tid

        # Regex splitter that catches special tokens intact (longest first)
        sorted_specials = sorted(self._special_str_to_id.keys(), key=len, reverse=True)
        self._special_pattern = re.compile(
            "(" + "|".join(re.escape(tok) for tok in sorted_specials) + ")"
        )

    # ------------------------------------------------------------------
    # Properties matching CNCMConfig
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __len__(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def unk_token_id(self) -> int:
        return 3

    @property
    def system_start_id(self) -> int:
        return 4

    @property
    def user_start_id(self) -> int:
        return 5

    @property
    def assistant_start_id(self) -> int:
        return 6

    @property
    def turn_end_id(self) -> int:
        return 7

    @property
    def think_start_id(self) -> int:
        return 8

    @property
    def think_end_id(self) -> int:
        return 9

    @property
    def tool_call_start_id(self) -> int:
        return 10

    @property
    def tool_call_end_id(self) -> int:
        return 11

    @property
    def tool_resp_start_id(self) -> int:
        return 12

    @property
    def tool_resp_end_id(self) -> int:
        return 13

    @property
    def tools_start_id(self) -> int:
        return 14

    @property
    def tools_end_id(self) -> int:
        return 15

    @property
    def fim_prefix_id(self) -> int:
        return self._fim_base + 0

    @property
    def fim_suffix_id(self) -> int:
        return self._fim_base + 1

    @property
    def fim_middle_id(self) -> int:
        return self._fim_base + 2

    @property
    def fim_end_id(self) -> int:
        return self._fim_base + 3

    @property
    def kw_start_id(self) -> int:
        return self._fim_base + 4

    @property
    def kw_end_id(self) -> int:
        return self._fim_base + 5

    @property
    def kw_empty_id(self) -> int:
        return self._fim_base + 6

    @property
    def mlen_1_id(self) -> int:
        return self._fim_base + 7

    @property
    def mlen_2_id(self) -> int:
        return self._fim_base + 8

    @property
    def mlen_3_id(self) -> int:
        return self._fim_base + 9

    @property
    def mlen_4plus_id(self) -> int:
        return self._fim_base + 10

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

    def format_fim_prompt(
        self,
        prefix: str,
        suffix: str,
        prefix_budget: Optional[int] = None,
        suffix_budget: Optional[int] = None,
    ) -> Tuple[str, List[int]]:
        """Format PSM (Prefix-Suffix-Middle) causal prompt with edge context budgets.

        Keep the `prefix_budget` tokens of `prefix` and `suffix_budget` tokens of
        `suffix` nearest the cursor (dropping from the far ends).
        """
        p_ids = self.encode(prefix, add_special_tokens=False)
        s_ids = self.encode(suffix, add_special_tokens=False)
        if prefix_budget is not None and prefix_budget > 0 and len(p_ids) > prefix_budget:
            p_ids = p_ids[-prefix_budget:]
        if suffix_budget is not None and suffix_budget > 0 and len(s_ids) > suffix_budget:
            s_ids = s_ids[:suffix_budget]

        psm_ids = (
            [self.fim_prefix_id]
            + p_ids
            + [self.fim_suffix_id]
            + s_ids
            + [self.fim_middle_id]
        )
        return self.decode(psm_ids, skip_special_tokens=False), psm_ids

    @property
    def special_token_ids(self) -> Tuple[int, ...]:
        return tuple(self._special_id_to_str.keys())

    # ------------------------------------------------------------------
    # Vocabulary inspection helpers (used by grammar constraints & debug)
    # ------------------------------------------------------------------
    def id_to_token(self, token_id: int) -> str:
        """Return display string for a token ID (special token or decoded bytes)."""
        if token_id in self._special_id_to_str:
            return self._special_id_to_str[token_id]
        raw = self._id_to_bytes.get(token_id)
        if raw is None:
            return self._special_id_to_str[self.unk_token_id]
        return raw.decode("utf-8", errors="replace")

    def id_to_bytes(self, token_id: int) -> bytes:
        """Return raw UTF-8 bytes for a token ID."""
        if token_id in self._special_id_to_str:
            return self._special_id_to_str[token_id].encode("utf-8")
        return self._id_to_bytes.get(token_id, b"")

    def token_to_id(self, token: str) -> Optional[int]:
        if token in self._special_str_to_id:
            return self._special_str_to_id[token]
        return self._bytes_to_id.get(token.encode("utf-8"))

    def get_vocab(self) -> Dict[str, int]:
        vocab: Dict[str, int] = dict(self._special_str_to_id)
        for tid, raw in self._id_to_bytes.items():
            try:
                text = raw.decode("utf-8")
                if text not in vocab:
                    vocab[text] = tid
            except UnicodeDecodeError:
                pass
        return vocab

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------
    def _bpe_encode_chunk(self, raw: bytes) -> List[int]:
        """Run exact learned BPE merge ranks on a byte chunk (falling back to byte IDs)."""
        assert self._bpe_merges is not None
        if not raw:
            return []
        if len(raw) == 1:
            return [_BYTE_OFFSET + raw[0]]

        # Pre-split into short word/whitespace/digit pieces so pure-Python BPE stays O(L) with caching
        if self._bpe_split_digits:
            segments = re.findall(rb"[ \t]*[A-Za-z_]+|[0-9]|\s+|[^ \t\n\rA-Za-z_0-9]+", raw)
        else:
            segments = re.findall(rb"[ \t]*[A-Za-z_0-9]+|\s+|[^ \t\n\rA-Za-z_0-9]+", raw)

        merges = self._bpe_merges
        cache = self._bpe_word_cache
        out_ids: List[int] = []
        for seg in segments:
            if len(seg) == 1:
                out_ids.append(_BYTE_OFFSET + seg[0])
                continue
            cached = cache.get(seg)
            if cached is not None:
                out_ids.extend(cached)
                continue
            symbols: List[bytes] = [bytes([b]) for b in seg]
            while len(symbols) >= 2:
                best_rank: Optional[int] = None
                best_pair: Optional[Tuple[bytes, bytes]] = None
                for k in range(len(symbols) - 1):
                    pair = (symbols[k], symbols[k + 1])
                    rk = merges.get(pair)
                    if rk is not None and (best_rank is None or rk < best_rank):
                        best_rank = rk
                        best_pair = pair
                if best_pair is None:
                    break
                merged: List[bytes] = []
                i = 0
                n_s = len(symbols)
                while i < n_s:
                    if i + 1 < n_s and (symbols[i], symbols[i + 1]) == best_pair:
                        merged.append(symbols[i] + symbols[i + 1])
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                symbols = merged

            seg_ids: List[int] = []
            for sym in symbols:
                tid = self._bytes_to_id.get(sym)
                if tid is not None:
                    seg_ids.append(tid)
                else:
                    for b in sym:
                        seg_ids.append(_BYTE_OFFSET + b)
            if len(cache) < 65536 and len(seg) <= 64:
                cache[seg] = tuple(seg_ids)
            out_ids.extend(seg_ids)
        return out_ids

    def _encode_bytes(self, data: bytes) -> List[int]:
        if self._bpe_merges is not None:
            return self._bpe_encode_chunk(data)
        ids: List[int] = []
        n = len(data)
        i = 0
        root = self._trie_root
        while i < n:
            node = root
            best_id = _BYTE_OFFSET + data[i]
            best_end = i + 1
            j = i
            while j < n:
                nxt = node.children.get(data[j])
                if nxt is None:
                    break
                node = nxt
                j += 1
                if node.token_id is not None:
                    best_id = node.token_id
                    best_end = j
            ids.append(best_id)
            i = best_end
        return ids

    def _encode_via_hf_fast(self, text: str) -> List[int]:
        assert self._hf_fast_tok is not None and self._hf_id_to_cncm_ids is not None
        enc = self._hf_fast_tok.encode(text, add_special_tokens=False)
        lut = self._hf_id_to_cncm_ids
        out: List[int] = []
        for hid in enc.ids:
            out.extend(lut[hid])
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode UTF-8 text into token IDs (preserving CNCM special tags)."""
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)

        if text:
            if self._hf_fast_tok is not None and self._hf_id_to_cncm_ids is not None:
                parts = self._special_pattern.split(text)
                if len(parts) == 1:
                    ids.extend(self._encode_via_hf_fast(text))
                else:
                    for part in parts:
                        if not part:
                            continue
                        sp_id = self._special_str_to_id.get(part)
                        if sp_id is not None:
                            ids.append(sp_id)
                        else:
                            ids.extend(self._encode_via_hf_fast(part))
            else:
                parts = self._special_pattern.split(text)
                for part in parts:
                    if not part:
                        continue
                    sp_id = self._special_str_to_id.get(part)
                    if sp_id is not None:
                        ids.append(sp_id)
                    else:
                        ids.extend(self._encode_bytes(part.encode("utf-8")))

        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs back to UTF-8 text."""
        out_chunks: List[str] = []
        byte_buf = bytearray()

        def _flush_bytes() -> None:
            if byte_buf:
                out_chunks.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for raw_tid in ids:
            tid = int(raw_tid)
            if tid in self._special_id_to_str:
                _flush_bytes()
                if not skip_special_tokens:
                    out_chunks.append(self._special_id_to_str[tid])
            else:
                b = self._id_to_bytes.get(tid)
                if b is not None:
                    byte_buf.extend(b)
                elif not skip_special_tokens:
                    _flush_bytes()
                    out_chunks.append(self._special_id_to_str[self.unk_token_id])

        _flush_bytes()
        return "".join(out_chunks)

    def batch_encode(
        self, texts: Sequence[str], add_special_tokens: bool = False
    ) -> List[List[int]]:
        if self._hf_fast_tok is not None and self._hf_id_to_cncm_ids is not None:
            lut = self._hf_id_to_cncm_ids
            # Fast multi-threaded Rust batch encode for texts without special tags, exact fallback for tagged ones
            plain_indices: List[int] = []
            plain_texts: List[str] = []
            results: List[Optional[List[int]]] = [None] * len(texts)
            for idx, txt in enumerate(texts):
                if not txt or self._special_pattern.search(txt) is not None:
                    results[idx] = self.encode(txt, add_special_tokens=add_special_tokens)
                else:
                    plain_indices.append(idx)
                    plain_texts.append(txt)
            if plain_texts:
                encodings = self._hf_fast_tok.encode_batch(plain_texts, add_special_tokens=False)
                for idx, enc in zip(plain_indices, encodings):
                    seq: List[int] = [self.bos_token_id] if add_special_tokens else []
                    for hid in enc.ids:
                        seq.extend(lut[hid])
                    if add_special_tokens:
                        seq.append(self.eos_token_id)
                    results[idx] = seq
            return [r if r is not None else [] for r in results]
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    # ------------------------------------------------------------------
    # Chat & Agentic Tool-Use Template
    # ------------------------------------------------------------------
    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Any]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format multi-turn messages + optional tool schemas into CNCM prompt text."""
        parts: List[str] = []
        has_system = any(m.get("role") == "system" for m in messages)

        tools_block = ""
        if tools:
            formatted_tools: List[Any] = []
            for t in tools:
                if isinstance(t, dict):
                    formatted_tools.append(t)
                elif hasattr(t, "to_schema"):
                    formatted_tools.append(t.to_schema())
                elif hasattr(t, "name") and hasattr(t, "description"):
                    formatted_tools.append(
                        {
                            "name": getattr(t, "name"),
                            "description": getattr(t, "description"),
                            "parameters": getattr(t, "parameters", {}),
                        }
                    )
                else:
                    formatted_tools.append(str(t))
            tools_json = json.dumps(formatted_tools, ensure_ascii=False, separators=(",", ":"))
            tools_block = f"\n<|tools|>{tools_json}</|tools|>"

        if not has_system and tools_block:
            parts.append(
                "<|system|>You are a helpful CNCM assistant. Use <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call> when a tool is needed."
                + tools_block
                + "<|end|>"
            )

        system_injected = False
        for msg in messages:
            role = str(msg.get("role", "user")).lower()
            content = str(msg.get("content", ""))
            if role == "system":
                extra = tools_block if (tools_block and not system_injected) else ""
                system_injected = True
                parts.append(f"<|system|>{content}{extra}<|end|>")
            elif role == "user":
                parts.append(f"<|user|>{content}<|end|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>{content}<|end|>")
            elif role == "tool":
                parts.append(f"<tool_response>{content}</tool_response>")
            else:
                parts.append(f"<|user|>{content}<|end|>")

        if add_generation_prompt:
            parts.append("<|assistant|>")

        return "".join(parts)

    # ------------------------------------------------------------------
    # Optional HuggingFace tokenizer.json loader & BPE trainer
    # ------------------------------------------------------------------
    @staticmethod
    def _gpt2_bytes_to_unicode_inv() -> Dict[str, int]:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = list(bs)
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {chr(c): b for b, c in zip(bs, cs)}

    @classmethod
    def _decode_hf_bpe_piece(cls, piece: str, byte_dec: Dict[str, int]) -> bytes:
        """Convert a ByteLevel-encoded HF BPE token piece back to exact raw UTF-8 bytes."""
        if all(ch in byte_dec for ch in piece):
            return bytes(byte_dec[ch] for ch in piece)
        cleaned = piece.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n").replace("ĉ", "\t")
        return cleaned.encode("utf-8")

    @classmethod
    def from_hf_json(
        cls,
        path: Union[str, Path],
        vocab_size: Optional[int] = None,
    ) -> "CNCMTextTokenizer":
        """Load subword tokens & optional BPE merges from a HuggingFace tokenizer.json file.

        Preserves all 16 core/agent special tokens (`0..15`), 256 byte-fallback tokens
        (`16..271`), and 11 PG-FIM high-tail special tokens (`vocab_size-11 .. vocab_size-1`).
        """
        file_path = Path(path)
        data = json.loads(file_path.read_text(encoding="utf-8"))
        model_section = data.get("model", {})
        raw_vocab = model_section.get("vocab", {})
        raw_merges = model_section.get("merges", [])

        special_strs = {tok for _, tok in SPECIAL_TOKENS} | set(PG_FIM_SPECIAL_STRINGS)
        byte_dec = cls._gpt2_bytes_to_unicode_inv()

        # Detect whether the HF JSON used a ByteLevel decoder/pre-tokenizer and/or Digits split
        dec_type = str((data.get("decoder") or {}).get("type", ""))
        pre_tok = data.get("pre_tokenizer") or {}
        pre_tok_str = json.dumps(pre_tok)
        is_byte_level = ("ByteLevel" in dec_type) or ("ByteLevel" in pre_tok_str)
        split_digits = '"Digits"' in pre_tok_str

        extracted: List[bytes] = []
        seen: set[bytes] = set()

        # Sort by HF vocab rank so most frequent merges enter first
        items = sorted(raw_vocab.items(), key=lambda kv: kv[1])
        for tok_str, _ in items:
            if tok_str in special_strs:
                continue
            raw = (
                cls._decode_hf_bpe_piece(tok_str, byte_dec)
                if is_byte_level
                else tok_str.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n").encode("utf-8")
            )
            if len(raw) >= 2 and raw not in seen:
                # Never allow multi-digit tokens when Digits(individual_digits=True) is active
                if split_digits and raw.isdigit():
                    continue
                seen.add(raw)
                extracted.append(raw)

        target_vocab = (
            int(vocab_size)
            if vocab_size is not None
            else max(_SUBWORD_OFFSET + _NUM_FIM_SPECIALS + len(extracted), 4096)
        )
        tok_obj = cls(vocab_size=target_vocab, custom_subwords=extracted)

        # Load exact BPE merge ranks if present so encode() uses real BPE merges
        if raw_merges:
            merges_map: Dict[Tuple[bytes, bytes], int] = {}
            for rank, entry in enumerate(raw_merges):
                if isinstance(entry, str):
                    parts = entry.split(" ", 1)
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    parts = [str(entry[0]), str(entry[1])]
                else:
                    continue
                if len(parts) != 2:
                    continue
                a_b = (
                    cls._decode_hf_bpe_piece(parts[0], byte_dec)
                    if is_byte_level
                    else parts[0].replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n").encode("utf-8")
                )
                b_b = (
                    cls._decode_hf_bpe_piece(parts[1], byte_dec)
                    if is_byte_level
                    else parts[1].replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n").encode("utf-8")
                )
                merged_b = a_b + b_b
                if merged_b in tok_obj._bytes_to_id:
                    merges_map[(a_b, b_b)] = rank
            if merges_map:
                tok_obj._bpe_merges = merges_map
                tok_obj._bpe_split_digits = split_digits

        # Attach native Rust `tokenizers.Tokenizer` fast path when installed (~200x faster encoding!)
        try:
            from tokenizers import Tokenizer as HFTokenizer  # type: ignore[import-not-found]

            hf_fast = HFTokenizer.from_file(str(file_path))
            hf_vocab = hf_fast.get_vocab(with_added_tokens=True)
            if hf_vocab:
                max_hid = max(int(v) for v in hf_vocab.values())
                lut: List[Tuple[int, ...]] = [() for _ in range(max_hid + 1)]
                for tok_str, hid in hf_vocab.items():
                    h_idx = int(hid)
                    sp = tok_obj._special_str_to_id.get(tok_str)
                    if sp is not None:
                        lut[h_idx] = (sp,)
                        continue
                    raw = (
                        cls._decode_hf_bpe_piece(tok_str, byte_dec)
                        if is_byte_level
                        else tok_str.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n").encode("utf-8")
                    )
                    if split_digits and len(raw) >= 2 and raw.isdigit():
                        lut[h_idx] = tuple(_BYTE_OFFSET + b for b in raw)
                    else:
                        cid = tok_obj._bytes_to_id.get(raw)
                        if cid is not None:
                            lut[h_idx] = (cid,)
                        else:
                            lut[h_idx] = tuple(_BYTE_OFFSET + b for b in raw)
                tok_obj._hf_fast_tok = hf_fast
                tok_obj._hf_id_to_cncm_ids = lut
        except Exception:
            pass

        return tok_obj

    @classmethod
    def train_bpe(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4096,
        save_path: Optional[Union[str, Path]] = None,
        min_frequency: int = 2,
    ) -> "CNCMTextTokenizer":
        """Train a corpus-specific Byte-Level BPE tokenizer preserving CNCM special tokens and single digits.

        Requires optional dependency `tokenizers` (`pip install tokenizers`).
        Uses `Sequence([Digits(individual_digits=True), ByteLevel(add_prefix_space=False)])`
        so digits `0..9` stay single-digit tokens for Python arithmetic and JSON numbers,
        and preserves all 16 core special tokens (`0..15`) plus 11 PG-FIM high-tail tokens
        (`vocab_size - 11 .. vocab_size - 1`).
        """
        if vocab_size < _SUBWORD_OFFSET + _NUM_FIM_SPECIALS:
            raise ValueError(
                f"vocab_size must be >= {_SUBWORD_OFFSET + _NUM_FIM_SPECIALS}, got {vocab_size}"
            )
        try:
            from tokenizers import Tokenizer as HFTokenizer  # type: ignore[import-not-found]
            from tokenizers import decoders, models, pre_tokenizers, trainers  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "CNCMTextTokenizer.train_bpe() requires the optional `tokenizers` package. "
                "Install it with: pip install tokenizers"
            ) from exc

        import tempfile

        all_specials = [tok for _, tok in SPECIAL_TOKENS] + list(PG_FIM_SPECIAL_STRINGS)
        hf_tok = HFTokenizer(models.BPE(unk_token="<|unk|>"))
        hf_tok.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )
        hf_tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=int(min_frequency),
            special_tokens=all_specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        hf_tok.train_from_iterator(iter(texts), trainer=trainer)

        if save_path is not None:
            out_file = Path(save_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            hf_tok.save(str(out_file))
            return cls.from_hf_json(out_file, vocab_size=int(vocab_size))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "cncm_bpe_tokenizer.json"
            hf_tok.save(str(tmp_file))
            return cls.from_hf_json(tmp_file, vocab_size=int(vocab_size))
