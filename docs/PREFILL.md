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

## The expert path, second try: decode without a slab (v7, `mm_gu7` / `mm_down7`)

The mpp path wins the drive budget by writing each expert to an fp16 slab once and reading it back: at TM 32 a
5120-row chunk writes 194 GB of slab and reads 388 GB of it back. The kernels that avoid that (v3-v6) decode the
pack's bytes straight into matmul2d's right-input cooperative tensor, which needs the M5's element layout: runs of 4
consecutive k at one n, so one fetch feeds four weights. On the M1 the same tensor holds pairs of n at one k, so a
lane's four values are four different rows, each with its own block header - the trick does not transfer, which is
why `STRATA_METAL_PROMPT_EXPERTS=v3` is off there.

v7 stages the decoded TK x TN block in **threadgroup memory** instead (as `mm_gu` / `mm_gu2` do) and lets matmul2d
read it from there, so nothing depends on the layout. It keeps v6's plan (`mm_scan6`): a tile is a group of up to MS
16-row subtiles of one expert, so the decode is paid once for 64 rows instead of once for 16. Nothing of the expert is
ever written to device memory: the pack's quantized bytes are the only traffic.

| | |
|---|---|
| parity (`moe_mm_parity --v7 4`, IQ3_XXS gate/up + Q5_0 down, 256 tokens) | plan ok, y vs the fp16-operand reference **rel L2 3.16e-05** (the v2/v3/v6 family's 2.6-3.2e-05), same output tokens as mpp end to end |
| harness, the engine's shape (512 experts, 5120 rows = 100 rows an expert, the pack's six type pairs) | **929-1008 tok/s experts-only** (gate/up 3.9-4.4 TFLOP/s, down 6.2-6.6); mpp in the same harness at 2048 rows: 616 |
| the engine, a 5120-row chunk (GPU ms of the MoE class) | v7 **11 658**, mpp **6 121**; with the drive faked (`STRATA_METAL_STREAM_FAKE=1`) 9 900 against 4 740 |
| `STRATA_METAL_PROMPT_MM_PROBE`: the same 12287-token prompt | full 75.1 s, without the decode 69.4 s, without the matrix multiply 36.1 s |

So the fused path is 1.4x faster than the slab path **in isolation** and 1.9x slower **in the engine**, and the probes
say why it is not the decode: removing the decode buys 5.7 s of a 75 s prompt, removing the MMA 39 s. The slab path
feeds one dynamic-K matmul from a dense, sequential fp16 buffer that the MMA loop never pauses for; v7 puts a
barrier and a threadgroup round trip every 32 k in front of the same accumulator registers. On this GPU (5.1 TFLOP/s
for mpp in the engine against 2.4 for v7) the slab is the cheaper way to keep the matrix units fed, which is the
opposite of the assumption the v3-v6 kernels were written on and the reason they were gated to the M5.

It stays in the tree as `STRATA_METAL_PROMPT_EXPERTS=v7`, off by default: it is the path for a GPU without the M5's
layout, it needs no slab space in the arena, and `STRATA_METAL_PROMPT_V7_TILE` sweeps its shape. What it does not do
is make an M1 Max's prompt faster.

## Where a chunk's GPU time goes, by class (mpp, TM 32, 5120 rows)

The classes are additive differences (`STRATA_METAL_PROMPT_SKIP`), so each is the class plus what depends on it:

| class | ms | share | measured as |
|---|---|---|---|
| routed experts (mpp TM 32, slab path) | 6 210 | 44% | chunk 13 907 - `SKIP=moe` 7 694 |
| GDN layers (36 of the 48) | 3 690 | 26% | chunk - `SKIP=gdn` 10 213 |
| QSA layers (12) | 3 140 | 22% | chunk - `SKIP=qsa` 10 772 |
| hc mixes | 1 170 | 8% | chunk - `SKIP=hc` 12 738 |

The attention machinery is now as large as the experts, and both are GEMM-dominated: GDN's fused input projection is
16 480 x 2 560 a layer and QSA's is 12 288 x 2 560 plus 6 144 x 2 560, so 36 + 12 layers of them are ~10 GFLOP a
token. At the 9.6 TFLOP/s of the matrix units that is 1.1 ms a token for the two, against the 2.9 ms a token they
take: they run at 60% of the ceiling, the same place the dense half has been all along.

## The dense projections' block order (`STRATA_METAL_PROMPT_GEMM_ORDER=sw`)

`gemm_f16` tiles Y as TM x TN and the host put N on the grid's x, so Metal (which issues threadgroups x-major) walked
the whole B matrix once per TM-row tile of A - 80 traversals of every dense projection in a 5120-row chunk. With M on
x and N on y, the blocks in flight share one N column block: its B slice (TN x K halfs = 320 KB at TN 64) is read
once and then served from L2, and A (M x K = 26 MB) becomes the operand that repeats - and it fits L2.

| order and tile | chunk GPU ms |
|---|---|
| N on x (the old order), 64_64_4 | 13 437 |
| **M on x (sw), 64_64_4** | **12 990** |
| sw, 128_64_4 | 14 091 |
| sw, 64_128_4 | 13 883 |
| sw, 128_128_4 | 17 908 |

-3.3% and bit-identical output, not the 1.4x the traffic model predicted: at 64-column blocks the B slice was mostly
L2-served already, so what the projections are short of is MMA issue and accumulator occupancy, not the drive. Taller
tiles make that worse, which is the same lesson as the fp16-accumulate probe: on this GPU the dense half's remaining
third is scheduling, not bandwidth.

## The per-kernel table of a chunk (STRATA_METAL_PROMPT_KPROF), and what it changes

`STRATA_METAL_PROMPT_KPROF=<path>` replays a chunk's dispatch list one dispatch at a time after it has run and writes
each kernel's GPU time (launch overhead included, so read it as a ranking; the answers afterwards are wrong). For a
5120-row chunk on the M1 Max, the production config (mpp TM 32, the swapped dense order):

| ms | n | kernel | efficiency |
|---|---|---|---|
| 1 866 | 384 | `mm_gu_mpp_32_128_4` (expert gate/up) | **8.8 TFLOP/s with the down GEMM = 92% of the matrix ceiling** |
| 1 624 | 228 | `qsa_attn_fp16` (sparse attention) | 1.9 TFLOP/s - **20%**, scalar FMAs, no matrix units |
| 1 071 | 37 | `gemm_f16_sw` g80x160 (GDN's q\|k\|v, N 10 240) | 9.3 TFLOP/s - 97% |
| 952 | 97 | `gemm_f16_sw` g80x40 (N 2 560: hc/o projections) | 7.5 |
| 891 | 384 | `mm_dn_mpp_32_64_2` (expert down) | (with the gate/up above) |
| 630 | 36 | `gemm_f16_sw` g80x96 (GDN's y/z output, N 6 144) | 9.2 - 96% |
| 602 | 96 | `gemm_f16_sw` g80x6 (N 384: beta/alpha/gate) | **1.6 - small-N waste** |
| 581 | 36 | `gdn_scan_v` (the recurrent scan) | - |
| 448 | 96 | `gemm_hc_up` | - |
| 412 | 12 | `gemm_f16_sw` g80x192 (QSA's q+gate, N 12 288) | 8.9 |
| 234 | - | `mm_expand_*` (the fp16 slab decode) | 2.4% of the chunk |
| 206 | - | `mm_gather`, `mm_combine`, plan | 2% |
| 213 | - | norms, epilogues, `dq_f16`, `act_f16` | 2% |
| 2 016 | 240 | `qsa_*` altogether | 32% of the chunk's ceiling |
| **10 626** | 4 271 | **sum of single-dispatch times** | the chunk as one command buffer: **13 439** |

Three findings, and they invert the earlier ranking:

1. **The expert GEMMs are done.** 24.2 TFLOP of gate/up/down in 2.757 s is 8.8 TFLOP/s, 92% of what `mm_bench`
   measures `matmul2d` can do at all. There is nothing left there - which is the answer to the slab-vs-fused question
   from both sides: the slab path's GEMMs are already at the hardware's limit, so only the decode could ever be
   attacked, and it is 234 ms of the chunk.
2. **The dense half is at 78%**, not the 66% the class split suggested: 30.2 TFLOP in 4 046 ms. The GDN's two large
   projections are at 96-97% - they are the best-shaped GEMMs in the model (M 5120 x N 10 240 / 6 144 x K 2 560). What
   is left of the dense side is the *small* projections (N 384 at 1.6 TFLOP/s) and the 64x64 tile's own overhead.
3. **Sparse attention is the one large kernel that is not on the matrix units.** 2.0 s of a 13.4 s chunk at 20% of the
   ceiling, in scalar FMA dots: per cell, 12 query heads each do a 256-term dot through `simd_sum` on the ALU.
   `qsa_flash_fp16` - the matmul2d version - already carries a non-COOP_K4 decode branch, so
   `STRATA_METAL_PROMPT_QSA=flash` runs on an M1; measured on a 12287-token prompt it is **no faster** (39.5 s against
   39.6 s of prompt loop), because its non-M5 path fills the right-input cooperative tensor one element at a time and
   that costs what the scalar dots cost. Making this phase fast needs a kernel that stages K/V tiles in threadgroup
   memory and keeps the selection's indirection out of the inner loop: the last large piece of work in the engine.

A chunk of 5120 rows is therefore 8.1 s of unavoidable matrix math (60.6 TFLOP at 9.6 TFLOP/s), 13.4 s as measured,
and ~1.4 s of that gap is the drive's own writes (the same chunk with `STRATA_METAL_STREAM_FAKE=1` is 12.5 s). The
practical ceiling of this engine on this Mac is therefore ~10-11 s a chunk = **470-510 tok/s**; production is at
**349 tok/s** steady state (a 19 688-token prompt, second request in the same engine, 56.4 s) and the remaining 20%
is the attention kernel and the small-N projections.

## The first prompt after a start costs ~20 s of setup

Measured on the live server: the first request of a process reports `prompt_ms 107.2 s` for 19 686 tokens while its
engine loop is 87.3 s; the same prompt as the second request reports **56.4 s** (349 tok/s) with a 0.2 s gap. The
19.9 s is one-time: `prompt_gemm_init` is already called after the arena fill, so it is the lazy per-kernel pipeline
creation of the first chunk's build plus the server's first-request work, not the prompt itself. It is why the first
prompt of a fresh server reads about half the steady-state rate, and it is worth warming (a startup pass that creates
the chunk's pipelines, without holding the 6.2 GB prompt region, which would cost the arena ~3 000 resident experts).

## Prefetch depth does not matter

`STRATA_METAL_PREFETCH=6` measured 12 960 ms of chunk GPU against 12 990 at the default 12: with 51% of the experts
resident the stream is already ahead, and the exposure is concentrated in the first chunk of a prompt (14.6 s of stall
in the second chunk of a 19 686-token prompt, none afterwards).

## Sparse attention: the two cheap fixes, both measured and rejected

`qsa_flash_fp16` - the matmul2d version of the selected-cell attention, the one the M5 runs - carries a
non-COOP_K4 decode branch, so `STRATA_METAL_PROMPT_QSA=flash` does run on an M1. Two attempts to make it pay:

| attempt | prompt loop, 12287 tokens |
|---|---|
| v1 (scalar dots, the M1 default) | 39 613 ms |
| flash, its own M1 decode | 39 514 ms |
| flash, the K block staged coalesced in threadgroup memory | 43 650 ms |

The third variant was written because the M1's right-input layout holds pairs of n at one k, so filling it one
element at a time looks like a 128-byte line per 2-byte value - 64 KB of lines for a 32-k block's 1 KB. It is 10%
*slower*: those lines are already reused in L1 across the lanes that need the same row, so the change only added
instructions and two barriers a step. That also explains why flash and v1 tie - both are bound by the same thing, and
it is not the cache.

What is left for this kernel is a decomposition change, not a tuning one: the current tile is one token x 12 heads x
64 cells, so every token re-reads the same 2051 selected rows (2 MB a threadgroup, 20 GB a layer) and writes a 33-chunk
split-K partials buffer (817 KB a token, 4.2 GB a layer, which `qsa_merge` then reads back). Blocking 16 tokens into
one tile would divide the first by 16, and a 256-cell chunk would divide the second by 4 - both need the partials
layout and `qsa_merge` to change with them.

## Two more measured and rejected (so they are not tried again)

Both target the largest and the worst-shaped kernels of the table above; both lose, and for the same reason - what
looks like a memory stall is really the arithmetic that follows it.

| change | 5120-row chunk |
|---|---|
| `qsa_attn_fp16`: its four cells' rows loaded together, so one row's L2 latency overlaps the others' | attention `1624 -> 1734 ms` (7% slower): a cell costs the twelve `simdgroup` reductions after its read, not the read |
| `gemm_f16_sw` at half height (tm 32) for a narrow output - N 384 is six N tiles, 480 threadgroups at 1.6 TFLOP/s | N 384 `602 -> 672 ms`, N 512 `140 -> 181 ms`: not an occupancy problem either |

What is left for sparse attention is a decomposition, not a tuning: block the 16 tokens a tile (the current tile is
one token x 12 heads x 64 cells, so every token re-reads the same 2051 selected rows - 2 MB a threadgroup, 20 GB a
layer) and widen the cell chunk to 256 (the split-K partials are 817 KB a token, 4.2 GB a layer, and `qsa_merge` reads
them back). Both move the partials layout with them. On a tensor-core kernel at 60% of the ceiling that would be ~0.6 s
instead of 1.85 s, i.e. 423 tok/s instead of 349; the floor of 470-510 also needs the dense half's remaining 22% and
the ~1.4 s a chunk that the streaming writes cost.

## A third attention variant, and the conclusion

| change | attention, 5120-row chunk |
|---|---|
| the twelve heads' dots computed before any is reduced (independent chains instead of one at a time) | **4036 ms** against 1624 - 2.5x slower: the twelve accumulators plus the PV ones spill |

So `qsa_attn_fp16` has been tried three ways on this GPU and each loses or ties: the matmul2d flash kernel (39514 against
39613 ms of prompt loop), the flash kernel with its K block staged coalesced (43650 ms), and the ILP restructure above.
Its cost is the per-cell reduction chain, 43 cycles a head, and it is at the local optimum of its structure. The
decomposition (16 tokens a tile, 256-cell chunks, tensor cores, partials layout and `qsa_merge` moving with them) is
the only lever left in this engine, worth ~1 s of a 13.4 s chunk, and it is a rewrite rather than a change.

Unchanged and re-verified after each rejection: a 5120-row chunk runs in **12 983-13 564 ms** of GPU time and the greedy
answer is the same `output : 5513` every time.

## An outside survey, checked against these numbers

A research pass (llama.cpp issues/PRs, flash-moe's I/O study, MoE-Infinity, SSD-LLaMA, oMLX's ANE work) ranked its
candidates on the premise that *compute is hidden behind the drive*, so its top three all cut bytes: layer-first
prefill, a threaded `pread`+`F_NOCACHE` read path, and an expert-contiguous pack. This engine's own measurements put
the premise itself in question, and each item lands differently:

| the survey's item | what this engine measures |
|---|---|
| "you are I/O bound: 60 GB a chunk at 5.3 GB/s" | the opposite: a 5120-row chunk is 13.4 s of GPU (kernels sum to 10.6 s) and needs ~45 GB of reads, i.e. **3.3 GB/s** - half the drive's 6.4 GB/s. Faking the stream (`STRATA_METAL_STREAM_FAKE=1`) saves only 1.4 s of the 13.4 |
| threaded `pread` + `F_NOCACHE`, no madvise hints | already the path: `F_NOCACHE`, 6 I/O threads, 6.4-6.8 GB/s measured, and `STRATA_METAL_PREFETCH` 6/12/24 makes no difference |
| expert-contiguous, 2 MB-aligned pack | already the pack: each expert's gate/up/down is one 2,380,800-byte block, read with one request |
| layer-first prefill (one expert sweep per *prompt*; llama.cpp PR #32082) | real, but bounded here: it can only recover the *exposed* I/O, ~1.4 s a chunk (the plan already reads 45 GB a chunk from a 56 GiB set, 26% of it resident). It also makes each expert's slab be re-read by more rows, which is the mpp path's own metric. Worth ~9% on a 3-chunk prompt for a rework of the whole prompt loop |
| dequantise in-kernel, never materialise fp16 experts | measured: the slab's GEMMs are at 92% of the ceiling and the decode is 234 ms of the chunk (the v7 work above) |
| ANE offload of the MLP (+9.9% on an M1 Max, oMLX) | the largest measured item on this exact chip, and the only one that would move prefill without touching a kernel - but it needs the private `AppleNeuralEngine.framework`, per-model fixed-shape banks and a 4 GiB window |
| SwiGLU fusion, split-K (+1.2-5.9% upstream) | consistent with what is left in the dense half's 22% |

Also checked and not a lever: the prefill planner's drive model. It prints "drive 1.50 GB/s (assumed)" because its
rotational-disk probe does not recognize this volume, but feeding it the real 6.4 GB/s (`STRATA_SSD_GBPS=6.4`) leaves
the plan identical (chunk 5120, ring 96) and the loop within noise (39 060 against 39 613 ms): what binds the plan is
the page cache's size against the 56 GiB expert set, not the drive's speed.

And what the attention kernel's probes say (§ above): removing the dots **or** removing the cache loads leaves it at
1 624 ms, so it is bound by neither - its cost is per-threadgroup overhead, which is why the scalar version, the
matmul2d version and the blocked decode all land within 10% of each other.

## The attention kernel: five variants, all measured

With the probes saying neither the dots nor the cache loads cost anything, the hypothesis was that these kernels are
bound by per-threadgroup cost, so fewer and fatter threadgroups should win. They do not:

| variant | attention |
|---|---|
| baseline (64-cell chunks, one head's dot reduced at a time) | **1 624 ms**, 0.317 ms a row |
| 128-cell chunks (17 threadgroups a token-kv instead of 33) | 0.428 ms a row - **35% worse** |

Together with the flash tie, the staged decode (10% worse) and the ILP restructure (2.5x worse), that is five
hypotheses measured and five rejections. What the numbers say is narrower than any of them: the kernel is 27 µs a
threadgroup for work that should be 2-4 µs, the two probes remove the only candidates that are *inside* the
threadgroup, and the per-threadgroup stages that remain (the initial Q loads from an fp32 buffer, the 12-head softmax,
the 12 KB partials write) are each only a few KB. The honest state is that this kernel's cost is not yet explained, and
that the next step is not another variant but a profile with `gpuStartTime`/`gpuEndTime` per dispatch and a
counter-based look at a single threadgroup, not a rewrite.

## The attention kernel's probes: what is free, and what that leaves

`STRATA_METAL_PROMPT_QSA_PROBE` removes one piece at a time from `qsa_attn_fp16` (wrong answers; function constant 5):

| removed | attention, 5120-row chunk |
|---|---|
| nothing (baseline) | 1 624 ms |
| the score dots (probe 1) | 1 638 |
| the K cache loads (probe 2) | 1 624 |
| the partials store (probe 3) | 1 634 |
| the softmax's exp (probe 4) | 1 635 |
| the V loads, four cells a pass | 1 760 (8% worse) |
| the twelve accumulators held live | 4 036 (2.5x worse) |
| 128-cell chunks instead of 64 | 0.428 against 0.317 ms a row |

Everything inside the threadgroup is free, and *no* variant of it is cheaper. With 16 896 threadgroups a dispatch and
7.1 ms a dispatch, the GPU is finishing one threadgroup every 420 ns and each threadgroup's code costs nothing - so
what this kernel pays is its own granularity: 256 threads and 15.5 KB of threadgroup memory each, which fits one or two
threadgroups per core, and nothing inside to hide the latency of the stages around them. The fix is therefore not a
variant of this kernel but a different shape of threadgroup (a simdgroup-sized one, ~2 KB, sixteen to a core) and the
Metal counter sampling to prove it before writing it. That is where this piece of the engine stops for now: the
attention is 1.85 s of a 13.0 s chunk, every cheaper variant has been measured and rejected, and the remaining work is
a kernel of a different granularity, not a tuning of this one.

## The end of what probing can say about the attention

The cross-lane reduction was the last stage that had a mechanism behind it: 3 840 `simdgroup` reductions a
threadgroup, sixty a cell, at a handful of cycles each is the right order for its bulk. Mapping a head onto **eight
lanes instead of thirty-two** (a lane holds 32 dims, the dot reduces in three shuffle steps instead of five) measures
**1 659 against 1 624 ms** - neutral, the registers it adds paying for the shuffles it saves.

With that, every stage of the kernel has been removed in turn and every one measures free: the dots, the K loads, the V
loads, the partials store, the softmax's exp, the cross-lane reduction, the threadgroup footprint (fp16 Q: neutral) and
the threadgroup granularity (128 cells: 35% worse; the matmul2d version: a tie). Nothing inside the threadgroup costs
anything, and nothing inside it can be made cheaper. That is not a tuning result, it is a statement that the kernel's
27 us a threadgroup is spent somewhere no source-level change reaches - which leaves Metal counter sampling
(`MTLCounterSampleBuffer`, per-threadgroup occupancy and stall counters) as the only way to find it, and no further
variants worth writing blind.

## The instrument that would have closed it does not exist on this Mac

The next step for the attention kernel was METAL counter sampling. A three-line probe of the device settles it:

```
device: Apple M1 Max
set timestamp : GPUTimestamp          # and nothing else
```

`MTLDevice.counterSets` on this machine offers the timestamp set alone - no stage utilization, no occupancy, no stall
counters (those arrive with later GPU generations). So the one measurement that could say where `qsa_attn_fp16`'s 27 us
a threadgroup goes is not available here, and neither is any other counter-based analysis of the eight variants that
were measured. The kernel stays as measured: 1 624 ms for a 5120-row chunk, every stage of it free when removed, and
that is the end of what this hardware allows anyone to learn about it without writing a new kernel of a different
granularity on speculation.

That also fixes the shape of the remaining work honestly: the gap between 349 tok/s and the 470-510 tok/s floor is
*not* reachable by tuning kernels on an M1 Max. It needs either the ANE offload (a private framework, per-model
fixed-shape banks) or layer-first prefill (a rework of the prompt loop), and both are subsystems rather than kernels.

## The GDN scan's four kernels measure the same

The last under-performing kernel with an available knob is the recurrent scan (581 ms, 9% of a chunk's kernels, and it
also runs once a decode step). `STRATA_METAL_GDN_SCAN` selects its variant; on a 5120-row chunk the four measure

| variant | chunk GPU |
|---|---|
| the chip-aware default | 13 159 ms |
| `scanr_2` | 13 136 |
| `scanr_4` | 13 092 |
| `scanr_8` | 12 944 |

all inside the +/-2% run-to-run spread this engine shows (12 983 - 13 564 ms over the session's runs), so there is no
lever in the choice. The scan's cost is structural: one threadgroup barrier a token, because a state column's
dot products reduce across the eight simdgroups that split the state's rows - which only a chunked delta-rule
rewrite (FLA's formulation: TB tokens a barrier, the intra-block part as a 16x16 triangular matmul) removes.

That closes the kernel-by-kernel audit: every kernel this engine dispatches has now been measured against the ceiling,
and for each one either reached it (experts 92%, GDN's large GEMMs 96-97%), or had its tile/variant/occupancy space
swept (dense 78%, scan level with itself), or could not be moved by any of eight source-level variants with no counter
instrument available on the GPU to go further (the attention).

## Layer-first prefill, measured out of contention

The one remaining route with an upstream precedent (llama.cpp PR #32082) was going to be justified by the drive
exposure. The engine's own stream accounting bounds that exposure much more tightly than the fake-stream experiment
suggested:

```
expert stream prompts 1 chunks: 14035 distinct lookups (292.4 per layer), in the arena 0.2802,
  streamed ahead 0.6949, missed 0.0249; 16278 streamed in; host 449.0 ms per chunk, of it waiting 429.7 ms
```

A 5120-row chunk looks up 292 of 512 experts a layer (57%, not the "essentially all of them" the survey assumed), gets
69% of them from the read-ahead, 28% from the arena and misses 2.5% - 350 urgent reads a chunk, ~1.2 ms each, which is
the whole of its blocking time. Widening the read-ahead does nothing about it:

| `STRATA_METAL_PREFETCH` | chunk total | chunk GPU | host waiting |
|---|---|---|---|
| 12 (default) | 13 764 ms | 13 177 | 130.8 ms |
| 64 | 13 644 | 13 017 | 143.4 |
| 512 (every expert of a layer) | 14 057 | 13 066 | 126.8 |

So what layer-first could recover is 130-430 ms a chunk, i.e. 1-3%, not the 9% the earlier estimate implied - and that is
not worth a rework of the prompt loop. (The 1.4 s that faking the stream saves is therefore not the waits: it is the
bandwidth and cache cost of the reads themselves, which no scheduling change removes.)

With that, both remaining routes are measured and neither is a kernel change: layer-first is worth 1-3%, and the ANE
offload (oMLX's +9.9% on this chip) is the only one above 5% and needs a private framework with per-model fixed-shape
banks. The engine stands at 349 tok/s against its 470-510 floor, and the floor itself assumes the drive's writes and
this kernel set.
