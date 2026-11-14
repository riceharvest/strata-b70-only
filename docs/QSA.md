# QSA: sparse attention for the dense Qwen3.8-27B on the B70

QSA (Qwen Sparse Attention) cuts the cost of the 16 full-attention layers at a long context. All keys and values are still stored. A small
learned indexer scores the context in blocks of 4 cells and each query row reads only its best 512 blocks (2,048 cells) and the cells of its
own block. The 48 GDN layers, the MTP block and the weights of the backbone do not change. Only 80 small tensors are new (52,434,944
parameters: per attention layer a q projection 5120 -> 4 x 128, a key projection 5120 -> 128 and three norm multipliers).

The code is `sycl/src/qwen35/q36_qsa.{hpp,dp.cpp}`; the engine calls it from sections marked `QSA BLOCK` in `q36_engine.dp.cpp`.
It is off unless `STRATA_Q36_QSA` names an indexer file. Nothing here has been measured on the card yet: the numbers a measurement must
give are in the parity program (`q36_mmvq_dense_parity --qsa --bench`).

## Get the indexers

    hf download pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF Qwen3.8-27B-QSA-GSQ-RCO-IQ3_S-mtp.gguf --local-dir DIR     # 12.3 GB, the retrained indexers
    python3 -I tools/qwen35/qsa_extract.py --src DIR/Qwen3.8-27B-QSA-GSQ-RCO-IQ3_S-mtp.gguf --out qwen38-27b-qsa-rco.f16.qsaidx --check

`qsa_extract.py` also reads `qsa-indexers.safetensors` of `higa2515/Qwen3.8-27B-QSA-BF16` (the original indexers). The file is 105 MB with
fp16 weights (the default; fp32 weights double it and the engine rounds them to fp16 on load).

## Run

    STRATA_Q36_QSA=/path/qwen38-27b-qsa-rco.f16.qsaidx strata ... --kv int8 ...

QSA needs the 8-bit cache with the transposed V (`--kv int8` and the matrix-engine attention chosen at start; leave `STRATA_Q36_XATT` and
`STRATA_Q36_KV_VT` alone) and one session. If it cannot run the engine prints `QSA requested ... but OFF: <why>` and runs dense.
`STRATA_Q36_QSA_PREFILL=1` also makes prompt chunks attend sparsely (default: the dense flash attention; the indexer keys are kept either way).
`STRATA_Q36_QSA_DUMP=<dir>` (with `--no-graph`) writes one decode step of one layer for `tools/qwen35/qsa_ref.py`.

## What a row does (the reference is kuraneko1/Qwen3.8-27B-QSA-llama.cpp)

    per cell     raw key = rms(k_proj . h) * k_pre                      (h: the layer's normed input)
    per block    when its 4th cell is processed: pooled = rms(mean of the 4 raw keys) * k_norm, rotated (NEOX, first 64 of 128 dims) to the block's first cell
    per row p    q_h = rope(rms(q_proj_h . h) * q_norm), h = 0..3
                 score(b) = sum_h relu(q_h . pooled_b) / sqrt(128) for every complete block b (4b + 3 <= p)
                 keep the 512 best (equal scores: the lower block index); a row with at most 512 complete blocks keeps all of them
                 attend to the kept blocks' cells and the cells of its own block up to p; a block is never counted twice

Selection is per row and shared by the 24 query heads of the layer. A window of rows (verify, MTP, a tree) selects per row, so a row of a
window is bitwise the row of a one-row step. A tree window ranks only the blocks wholly before its root and reads the window's cells under
its ancestor mask (the reference has no trees: this rule is ours).
