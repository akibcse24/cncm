"""Quick CPU throughput probe for nano Stage 1 / Stage 2 steps (used to size the local experiment budget)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cncm.config import get_preset
from cncm.text_tokenizer import CNCMTextTokenizer
from cncm.trainer import PackedChunkDataset, make_pg_fim_stage1_corpus, train_stage1, train_stage2

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
cfg = get_preset("nano", enc_layers=1, matryoshka_cutoffs=(4, 6, 7, 8), use_copy_decoder=True)
tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
docs = make_pg_fim_stage1_corpus(num_samples=400, seed=1)
s1 = PackedChunkDataset(docs, tok, chunk_size=cfg.chunk_size)
s2 = PackedChunkDataset(docs, tok, chunk_size=cfg.chunk_size, prefix_len=64)
t = time.perf_counter()
r = train_stage1(cfg=cfg, tokenizer=tok, dataset=s1, steps=6, batch_size=64, lr=2.5e-3, device="cpu", seed=1, verbose=False)
print(f"stage1 s/step={(time.perf_counter()-t)/6:.2f}")
s1m = r["model"].eval()
for mode in ("exact_ar", "joint"):
    t = time.perf_counter()
    train_stage2(cfg=cfg, tokenizer=tok, concept_tokenizer=s1m, dataset=s2, mode=mode, steps=4, batch_size=32, lr=1e-3, device="cpu", amp_dtype="fp32", seed=1, verbose=False, num_workers=0)
    print(f"stage2 {mode} s/step={(time.perf_counter()-t)/4:.2f}")
for loops in (3, 1):
    c = get_preset("nano", num_loops=loops)
    print("params nano loops", loops)
