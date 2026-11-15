# Prefill on the M1 Max: where the time goes, and what it costs to halve it

Engineering note, measured on this Mac (M1 Max, 32-core GPU, 64 GB, macOS 27) with the
UD-IQ4_XS-huihui pack of Qwen3.8-Flash-Next (48 layers, 512 routed experts a layer, 56.2 GiB of
experts, 2.46 GiB dense; 8 experts routed a token) served by the streaming Metal engine
(`--mmap-experts`, 32 GB arena, `--kv fp16`, MTP drafter on). Every number below is from the
engine's own trace (`STRATA_METAL_TRACE=1`, the per-chunk line and the per-class skip arms),
not an estimate, and the arms were run one per process with the previous engine's 47 GB and
its wired arena gone before the next started.

## The budget of one prompt chunk (5120 rows, warm arena)

| phase | ms | share | what bounds it |
|---|---|---|---|
| GPU: routed experts (mpp, 16-row tiles) | 6 900 | 40% | fp16 slab traffic + the expand; 0.8 TFLOP/s of useful math - 9% of the GPU's ceiling |
| GPU: dense projections + attention (GDN, QSA, hc, head) | 5 000 | 29% | the matrix units; 6.3 TFLOP/s - 66% of the 9.6 the GPU can do |
| GPU: QSA attention over the chunk | 1 500 | 9% | incremental |
| PLE rows: 16 n-gram rows a token from the 61 GB table | 3 100 | 18% | 27.7k random 4 KB reads at 116 MB/s - **IOPS, not bandwidth** |
| MTP drafter's K/V rows | 650 | 4% | one layer, run serially after the chunk |
| SSD: the expert deficit (~29 GB a chunk) | 0 (hidden) | 0% | overlapped by the layer-ahead stream; the chunk is 96.6% GPU busy |
| **chunk wall** | **17 100** | | 5120 rows = 2.9 ms a row = ~345 tok/s |

Two facts follow from the table and they decide everything else.

1. **The dense half is two thirds of the way to the hardware's ceiling.** `mm_bench` measures Metal's
   `matmul2d` at **9.61 TFLOP/s** for the 64x64x4 tile on a 4096^3 product, 9.45 on the prompt's qkv
   shape (2048x4096x2560). The dense projections move 4.8 Gparams a token (9.6 GFLOP) and the
   matrix work plus attention takes 6.3 s of the chunk: **6.3 TFLOP/s, 66%**. An
   **fp16-accumulating** twin of the same kernel (`mpp_mm_hacc_half_64_64_4`, added to `mm_bench`)
   measures **9.64 TFLOP/s**: on this GPU the matrix units do not go faster for a lower-precision
   accumulator, so the missing third is scheduling, the many small projections (router N=1, hc
   mixes N<=64, the N=1 gate) and the elementwise epilogues - not a faster instruction.
2. **The routed experts are ten times off.** 4.7 TFLOP of expert math takes 5.4-6.9 s: 0.8 TFLOP/s.
   They are bound by the fp16 slabs mpp materialises (194 GB written and 388 GB read a chunk at
   TM 32), not by math and not by the drive.

## The ceiling this puts on prefill

Useful math a prompt token costs ~13 GFLOP (dense 9.6 + routed 1.15 + attention). At the GPU's
9.6 TFLOP/s the *whole* prompt cannot run faster than 13 GFLOP / 9.6 TFLOP/s = **1.35 ms a token
= 740 tok/s**, and that assumes both halves reach the ceiling and every host phase vanishes.
Today's 345 tok/s is 47% of that ceiling. Halving the time therefore needs the experts at ~4x
(0.8 -> 3.5 TFLOP/s: 6.9 s -> 1.6 s) **and** the dense half at the ceiling (6.3 -> 9.6 TFLOP/s:
5 s -> 3.3 s) at the same time:

| | today | doubling needs |
|---|---|---|
| expert path | 6.9 s (0.8 TFLOP/s) | 1.6 s (3.5 TFLOP/s) |
| dense + attention | 5.0 s (6.3 TFLOP/s) | 3.3 s (9.6 TFLOP/s) |
| PLE gather, drafter, serve overhead | 0.65 + 3.1 + ~1.1 s | ~0 |
| chunk of 5120 rows | 17.1 s (345 tok/s) | 8.5 s (690 tok/s) |

So **the 400 GB/s of DRAM is not what limits prefill on this Mac - the 9.6 TFLOP/s of matrix units
is**, and doubling means collecting *every* item on that list, not one big trick. The expert path
is the only place where a single change buys more than half of it.

## What other engines do about it, and which parts apply here

* **llama.cpp, offloaded MoE prefill (issue #25859, PR #21067).** Their finding is exactly this
  engine's situation: "at prefill batch sizes the selection isn't buying anything... the union of
  experts selected across the batch is essentially all of them", so they skip the routing readback
  and stream each layer's whole expert set on a second stream, and they page-lock the offloaded
  weights so the copies DMA instead of bouncing through a pageable buffer. Measured +64% prefill
  (1143 -> 1880 tok/s at pp2048) on a 3060. Our engine already does both halves of that idea
  (`at_layer` streams all of layer l+1 behind layer l's compute; the arena is `mlock`ed), which is
  why the expert deficit (29 GB a chunk) costs nothing. What they do *not* do and we do is
  materialise the expert in fp16 first: llama.cpp's MMQ kernels dequantize inside the dot product,
  so their expert pass pays the quantized bytes once and no slab. That is item 1 below.
* **The drive queue.** Random uncached reads on this machine's SSD, measured at expert size:
  2.4 MB at queue depth 1 = 2.6 GB/s, QD4 = 5.4, QD8 = 5.6, QD16 = 8.7. The arena fill already
  runs at 6.5-6.8 GB/s with 6 threads, and 12 and 24 threads measured no better, so the stream is
  at the top of its curve and raising it is not a lever.
* **Bigger prompts per pass** (the ZINC write-up's and llama.cpp's `-ub`/`-b` sweeps both report
  chunk size as the biggest single knob for prefill). Here it is not: from 4096 to 8192 rows the
  per-row GPU time goes from 2.57 to 2.76 ms (the prompt buffers grow and the arena yields slots,
  so the extra rows buy nothing). The chunk-size win exists only while a prompt is drive-bound, and
  this one is not (96.6% GPU busy).
* **Precision tricks** (fp8/int8 matmul, fp16 accumulate). No int8/FP8 matrix path exists on an M1
  GPU, and fp16 accumulation measures the same 9.6 TFLOP/s as fp32 (see above), so this family is
  closed on this hardware.


## What changed here (measured)

| change | effect |
|---|---|
| `STRATA_METAL_PROMPT_MOE_TM` (default now 32 when a chunk gives an expert >= 24 rows) | a tile reads its expert's gate/up slab in full, so TM 32 halves the slab reads a 16-row tile makes. 4095 rows, GPU only: 12 386 -> 10 513 ms (-15%). TM 64 (one read) measured 11 203: its 64x128 tile is exactly the 32 KB threadgroup memory and loses occupancy |
| the PLE row gather pipelined (`STRATA_METAL_PLE_PREFETCH=0` restores the old behaviour) | a chunk's 16-rows-a-token gather (3.1 s of IOPS-bound random 4 KB reads) now runs on a helper thread during the *previous* chunk's GPU work, and lands in a pre-gathered buffer the chunk copies in (10 ms). chunk wall 16 966 -> 14 324 ms (-15.6%); the gather itself costs 4.4 s of wall inside the chunk and overlaps it |
| `STRATA_METAL_PROMPT_GEMM_TILE` (default `64_64_4`, unchanged) + 8-simdgroup and narrow-N instantiations | for measuring the dense half's tiles; 128x64x8 and friends are instantiated for the small projections too. Not faster here (the 64x64 tile is at the ceiling), kept as the knob |
| per-chunk trace: `... ms total (GPU ... ms, N dispatches; prologue ... = embeddings + PLE + build)` and `chunk ... ms, its drafter rows ... ms` | the prologue and the drafter are now separable from the GPU; that is how the 3.1 s PLE and the 0.65 s drafter were found |
| `STRATA_PLE_STATS=1` (engine side) | the gather's phases: issue, collect (blocked), decode, reads, dedup, cache hits, bytes. A 5120-token chunk is 81 920 rows, 85 527 page reads, 358 MB, 3.1 s |
| `mpp_mm_hacc_*` in `mm_bench` | the fp16-accumulate probe that rules the cheapest 2x out |

End to end on the engine with a 12 288-token prompt, chunk 5120: 50.4 s (248.8 tok/s) before the
two changes, 46.0 s (266.9 tok/s) after.

Through the live server (port 8180, `--mmap-experts`, the config's env), same 11.6k-token prompt,
one pair per config, greedy 3 tokens:

| | chunk GPU (4096 rows, 2nd chunk) | chunk total | request prompt_ms | tok/s |
|---|---|---|---|---|
| TM 16, PLE inline | 12 742 ms | 12 802 ms | 54 959 ms | 211.3 |
| TM 32, PLE pipelined | 11 366 ms (-10.8%) | 11 433 ms | 52 137 ms | 222.8 |

Two more runs of each side landed at 244 (old) and 247-254 (new) tok/s: the GPU's own run-to-run
spread on this box is +-10%, so read the chunk GPU column (-11%) rather than the request total,
which also carries the items below.

**The request is not only the chunks.** With `STRATA_METAL_TRACE=1` the engine reports the loop:
`prompt loop 11609 rows in 38113 ms` against the server's `prompt 11614 tokens ... in 47050 ms`.
The chunks plus the drafter's rows account for all 38.1 s, so about **9 s of every served prompt
(19%) happens outside the engine's prompt loop**, and it is there in both configs. That is the
next thing to measure, and it is worth more than the PLE gather was: the serve path's own prompt
preparation, the conversation cache's work around it and the state checkpoint are the candidates.

A/B discipline on this box: with `--mmap-experts` a prompt's answers are **not** reproducible run
to run, because which experts are in the arena when a layer runs depends on the streaming
timing, and a resident expert and a streamed one round slightly differently (the engine warns
about this at startup). Two runs of one unchanged binary disagreed at the third greedy token.
So prompt-level output equality proves nothing here. The mpp retile was checked where it can be
checked - `moe_mm_parity` against its fp16-operand reference, which is now tile-parameterised:

```
moe_mm_parity --gu IQ3_XXS --dn Q5_0 --mpp --tile 16,128,0,4 --T 512   # yhash 651a90905465dda3
moe_mm_parity --gu IQ3_XXS --dn Q5_0 --mpp --tile 32,128,0,4 --T 512   # yhash 651a90905465dda3
moe_mm_parity --gu IQ3_XXS --dn Q5_0 --mpp --tile 64,128,0,4 --T 512   # yhash 651a90905465dda3
```

All three heights hash identically and pass (rel L2 2.6e-05 against the fp16 reference), so the
retile changes the schedule and not a single value. (Before this change the test named
`mm_gu_mpp_16_128_4`/`mm_dn_mpp_16_64_2` whatever tile it was given, which made a TM 32 run look
like a 63% error - it was the 16-row kernel reading a 32-row plan.)

## What is left, in order of measured value

1. **The expert path's slabs (-2.5 to -3.5 s a chunk, ~20%).** mpp decodes each expert to fp16
   (3.3x the quantized bytes) and then reads that slab 1-2.5 times. The fused designs that avoid
   materialising it exist in `moe_mm.metal` (`mm_gu3`/`mm_gu4`/`mm_gu5`) but are **COOP_K4-only**
   (the M5's right-input layout): `STRATA_METAL_PROMPT_EXPERTS=v3` aborts this M1 Max's engine
   with `kIOGPUCommandBufferCallbackErrorInnocentVictim`, so an M1 twin of `mm_gu4` (decode once,
   apply to MS 16-row tiles) has to be written against the layout M1 exposes. That is the single
   largest remaining item and it is a kernel project, not a knob.
2. **The dense half's non-matrix parts (~1-1.5 s, ~10%).** 8.6 of the 9.6 TFLOP/s ceiling is
   matrix work; the remaining second is the many small projections (router N=1, hc mixes N<=64),
   the per-chunk dequantize of 2.46 GiB of dense weights, and the elementwise epilogues. Batching
   the small ones into one dispatch and keeping the dequantized copies across chunks of one
   request are both contained changes.
3. **The drafter's prompt rows (0.65 s a chunk, 4%).** `mtp_prefill` runs one command buffer per
   chunk, serially, because the residual rows it reads are overwritten by the next chunk. Copying
   those rows into a second buffer (`pb_.R2` exists for the window path; the chunked path does not
   double-buffer `p.R`) lets it run concurrently with the next chunk, and batching the 8-row
   groups into a chunk-wide attention pass would cut the work itself.
4. **The first chunk's PLE gather (3 s of a request, 0.6 ms a token).** It is the only gather
   still inline: the tokens are known when the request arrives, so a serve-side pre-gather at
   request parse time would hide it behind the conversation-cache lookup.
5. **Route-tail skipping for prompts.** An expert that every token of the chunk routes to at rank
   >= R contributes almost nothing and costs the same as a hot one. `STRATA_ROUTE_TAIL_SKIP`
   exists on the CUDA/HIP path only. At 8-of-512 routing the tail is ~10-15% of the expert math
   and ~10% of the streamed bytes; it changes answers (the CUDA path's mean KL was 0.038 at R=7),
   so it is an opt-in, and it is small next to items 1-3.
6. **Not on this list: fp16 accumulation, bigger chunks, more I/O threads, the page cache.**
   Measured and rejected: fp16 accumulate is the same 9.6 TFLOP/s; chunk 8192 and 10240 are
   slower per row than 5120 (bigger buffers, more arena slots held, no gain in the GPU); 12 and 24
   I/O threads change nothing (the drive is already fed); the expert file is read `F_NOCACHE` on
   purpose and the 53.5 GiB file tier does not fit the 42-47 GiB of page cache.

## Reproducing any line of this

```
# the chunk budget, the prologue split and the drafter's share
STRATA_METAL_TRACE=1 ... --tokens-file tokens.txt --max-new 2
# the classes (GPU only: the streaming reads are faked; the answers are wrong by design)
STRATA_METAL_STREAM_FAKE=1 STRATA_METAL_PROMPT_SKIP=moe|qsa|gdn
# the PLE gather's phases
STRATA_PLE_STATS=1
# the matrix ceiling and the fp16-accumulate twin
MMB_ONLY=mpp_mm_  ./build-metal/mm_bench     # 9.6 TFLOP/s fp32-accumulate
MMB_ONLY=hacc     ./build-metal/mm_bench     # 9.64 TFLOP/s fp16-accumulate
```
