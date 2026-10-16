# Task 12 report: Lab 2 a2 CK-Tile vector add

## Built
lab2-cktile-intro/a2-ck-vector-add/{ck_vector_add.hip (skeleton), test.cpp, README.md};
solutions/lab2-cktile-intro/a2-ck-vector-add/{ck_vector_add.hip, mutants/m1_single_p_dim, m2_origin_missing_tile, m3_single_block_loop}.
Skeleton: only `va` + `(void)va;`, everything else TODO. Grade output: 0/12 (all FAIL with NaN).

## Calibration (ratio = metric / in-process copy GB/s; GPUs 1 and 5, idle, 4 runs each)
- solution: gpu1 2881-2983 GB/s (copy ~1970-1996) -> ratio 1.45-1.50; gpu5 3066-3116 (copy ~1884-1904) -> 1.61-1.65. S_min = 1.45.
- m3 (grid=1 loop): 25.7-25.8 GB/s -> ratio 0.013 (N_max). S/N ~ 110 > 10 -> rule gives floor2(1.45/1.67)=0.86.
- Alternative designs (min over runs, ratio): kVec=1: 1654-1670 GB/s -> 0.84 (gpu1) / 0.885 (gpu5);
  kVec=2: 2273-2349 -> 1.17+; kVec=8: 2790-2864 -> 1.45+; kWarps=1: 3658-3763 -> 1.9+; kWarps=2: 3585-3713 -> 1.8+.
- 0.86 would fail kVec=1 (0.84), so lowered: kAddRatio = 0.64. Margins: solution 2.27x, kVec=1 1.31x, m3 49x (>=1.5 over N_max).
- Spot grade of kVec=1/kWarps=1/kVec=8 on gpu1 and gpu5 with final threshold: all 12/12 (kVec=1 1771 and 1661 GB/s).
- Note: ratio >1 for solution because 3 x 256MB traffic hits MALL / D2D copy counts differently; ratio is still stable.

## Verify (make verify, lab2-cktile-intro)
a1 OK; a2: solution 12/12 bandwidth=3554.3GB/s; m1 caught by one_tile, many_tiles, ragged, large, bandwidth (tiny passes by n<=256 happenstance);
m2 caught by many_tiles, ragged, large, bandwidth; m3 caught by bandwidth only (as designed); skeleton 0/12. VERIFY OK.

## Deviations / concerns
- Threshold 0.64 per controller rule (not brief's 0.60 placeholder) and kAddRatio comment documents calibration.
- m3 is the only mutant caught solely by the perf case; if perf were dropped it would pass (acceptable per brief; >=2 other mutants are deterministic).
- m2 passes one_tile/tiny (blockIdx 0 only), expected.
- README Hint 2 cites tile_distribution.hpp lines 97-105 (verified in ROCm 7.2). Cites tutorial sections §3.1-3.4 as briefed.
- Bandwidth metric printed for failing runs/skeleton is meaningless (empty kernel).

## Fix round 0 (controller ruling)
kAddRatio 0.64 -> 0.55 (kVec=1 ~1.53x, solution ~2.6x, m3 ~42x); test.cpp calibration comment expanded; README perf row updated (55% + D2D not a hard ceiling note). Command: cd lab2-cktile-intro && make verify
```
   mutant      m1_product_zero_init.hip         被抓住: e1_product
   mutant      m2_strides_colmajor.hip          被抓住: e2_row_major_strides
   mutant      m3_vec_width_bits.hip            被抓住: e4_if_constexpr
   skeleton    0/13
== a2-ck-vector-add
   solution    满分 12/12 bandwidth=3428.6GB/s
   mutant      m1_single_p_dim.hip              被抓住: one_tile, many_tiles, ragged, large, bandwidth
   mutant      m2_origin_missing_tile.hip       被抓住: many_tiles, ragged, large, bandwidth
   mutant      m3_single_block_loop.hip         被抓住: bandwidth
   skeleton    0/12

VERIFY OK
```
