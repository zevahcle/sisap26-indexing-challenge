#!/usr/bin/env python3
"""
SISAP 2026 — search.py  (TIRA contract)

    search.py --input <benchmark.h5> --task-description <config.json> --output <dir>

Task identity, k, HDF5 keys, sparse flag all come from config.json (NOT flags).
Drop-in replacement for sisap26-python-baseline/search.py: same load_task_config /
store_results helpers and the same "one .h5 per hyperparameter config" sweep pattern,
with the baseline's FAISS bodies replaced by our pipelines.

KEY CORRECTNESS FIXES vs our run_pynndescent.py dev script
----------------------------------------------------------
1. SELF IS INCLUDED in the output.  The official allknn/knns ground truth has the
   point itself at column 0 (1-based), and eval.py does a raw intersect of the first
   k columns with NO self-removal.  We therefore emit [self, top-(k-1) non-self],
   1-based.  (Emitting non-self caps recall at (k-1)/k and pushes ~0.82 below 0.80.)
2. FWHT pads to the next power of two, so non-power-of-2 inputs (e.g. the d=384
   gooaq spot-check used for the TIRA dry-run) project instead of crashing.
3. Memory-safe at 6.35M: stream-project, native PyNNDescent per oversample with a
   candidate-matrix disk handoff, then a single f16 reload for the rerank phase.
   Per-config buildtime/querytime are recorded honestly (shared projection + that
   config's own native build; rerank as querytime).
4. Optional --proj-method pca: PCA via TruncatedSVD (no centering, cosine-friendly),
   FIT ON A RANDOM SAMPLE (default 300k rows, stratified blocks) then transform all
   rows. Fit cost scales with the SAMPLE size (the saving); transform scales with N*p.
   Per-config buildtime includes the fit, so the reported end-to-end time is honest.
   --pca-drop-top discards the top-k components (the embedding "common direction" trick).
"""
import argparse, os, sys, time, gc, json
from pathlib import Path
import numpy as np
import h5py

# ----------------------------------------------------------------------
# Task 1 sweep grid  (<= 15 configs).  Edit here.  Grouped by proj so the
# projection is computed once per proj and reused across its oversamples.
# Recall predictions (dev 6.35M, from the trim anchors) are comments only.
# ----------------------------------------------------------------------
GRID_TASK1 = [
    # p=320, delta=0.01 (the sweep showed higher delta gives NO build saving at p=320
    # -- times were jitter-dominated -- and slightly lower recall, so 0.01 wins). One
    # build at n_neighbors=49; the ladder is free slices. Dev recall from the p=320
    # sweep: os=40 0.833, os=44 0.845, os=48 0.855 -> eval (~0.03 gap) ~0.803 / ~0.815
    # / ~0.825. The eval takes the fastest rung that clears; os=48 is the safety net.
    # (os=36, 0.818 dev, dropped -- it would not clear the test gap.)
    dict(proj=320, oversample=40, quantize="int8"),
    dict(proj=320, oversample=44, quantize="int8"),
    dict(proj=320, oversample=48, quantize="int8"),
]
# Total-run-time scoring => every rerank above counts. If you want to shave more,
# drop to the fewest rows that still clear with margin (e.g. 46/48/50, or even a
# single os=48 at 0.826). Bigger build lever: n_iters 12->8 is ~25% faster but can
# cost ~0.01 recall (os=42 could dip under 0.80; os=48/50 hold) — verify with
# recall_check.py before trusting it.

# PCA experiment grid (used by --proj-method pca). Sweeps projection dim at a fixed
# oversample to isolate the projection effect, plus a low-dim x oversample probe to
# see whether more candidates rescue aggressive PCA. Compare each row's recall to the
# random-256 baseline at the matching oversample from your earlier FWHT sweep.
GRID_TASK1_PCA = [
    dict(proj=32,  oversample=60, quantize="int8"),
    dict(proj=48,  oversample=60, quantize="int8"),
    dict(proj=64,  oversample=60, quantize="int8"),
    dict(proj=96,  oversample=60, quantize="int8"),
    dict(proj=128, oversample=60, quantize="int8"),
    dict(proj=192, oversample=60, quantize="int8"),
    dict(proj=256, oversample=60, quantize="int8"),   # head-to-head vs random-256/os=60
    # does more oversample rescue low-dim PCA?
    dict(proj=32,  oversample=80,  quantize="int8"),
    dict(proj=32,  oversample=100, quantize="int8"),
    dict(proj=64,  oversample=80,  quantize="int8"),
    dict(proj=128, oversample=80,  quantize="int8"),
]

# FWHT-JL projection-dimension sweep for --measure-ceiling. Finds the smallest p
# whose projected-space pool still contains the true neighbours, WITHOUT paying a
# NN-descent build per p. Read the ceiling per (p, oversample); the real pipeline
# recall is this minus PyNNDescent's loss (~0.02-0.03 at p=256 from the dev runs),
# so aim for a ceiling comfortably above 0.80 at the oversample you'd ship.
GRID_TASK1_CEILING = [
    dict(proj=p, oversample=os, quantize="int8")
    for p in (128, 160, 192, 224, 256)
    for os in (42, 46, 50)
]

N_TREES   = 10
N_ITERS   = 12
SEED      = 42
ALGO      = "fwht-jl+pynndescent+rerank"

# ======================================================================
# Baseline-compatible helpers (verbatim contract)
# ======================================================================

def load_task_config(path):
    with open(path) as f:
        return json.load(f)

def _resolve_key(cfg_key):
    """config 'data' / 'gt_I' may be a slash-path or a list path."""
    return cfg_key if isinstance(cfg_key, list) else cfg_key

def store_results(dst, algo, dataset, task, D, I, buildtime, querytime, params):
    os.makedirs(Path(dst).parent, exist_ok=True)
    with h5py.File(dst, "w") as f:
        f.attrs["algo"]      = algo
        f.attrs["dataset"]   = dataset
        f.attrs["task"]      = task
        f.attrs["buildtime"] = float(buildtime)
        f.attrs["querytime"] = float(querytime)
        f.attrs["params"]    = params
        f.create_dataset("knns",  I.shape, dtype=I.dtype)[:]  = I
        f.create_dataset("dists", D.shape, dtype=D.dtype)[:]  = D

# ======================================================================
# Numba rerank kernels (skip self via c == gi -> top-k NON-SELF neighbours)
# ======================================================================
try:
    from numba import njit, prange
    _NUMBA = True
    _LUT = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)

    @njit(parallel=True, fastmath=True, cache=True)
    def _kernel_u16(X_u16, cand, lut, qoff, K, knn_out, d_out):
        nl = cand.shape[0]; N = cand.shape[1]; d = X_u16.shape[1]
        for i in prange(nl):
            gi = i + qoff
            ids = np.full(K, np.int32(-1)); ds = np.full(K, np.float32(np.inf)); worst = np.float32(np.inf)
            for j in range(N):
                c = cand[i, j]
                if c == gi or c < 0: continue
                dot = np.float32(0.0)
                for l in range(d): dot += lut[X_u16[gi, l]] * lut[X_u16[c, l]]
                dist = np.float32(1.0) - dot
                if dist < worst:
                    p = K - 1
                    while p > 0 and dist < ds[p-1]:
                        ds[p] = ds[p-1]; ids[p] = ids[p-1]; p -= 1
                    ds[p] = dist; ids[p] = c; worst = ds[K-1]
            for j in range(K): knn_out[i, j] = ids[j]; d_out[i, j] = ds[j]

    @njit(parallel=True, fastmath=True, cache=True)
    def _kernel_int8(Xq, inv, cand, qoff, K, knn_out, d_out):
        nl = cand.shape[0]; N = cand.shape[1]; d = Xq.shape[1]
        for i in prange(nl):
            gi = i + qoff; si = inv[gi]
            ids = np.full(K, np.int32(-1)); ds = np.full(K, np.float32(np.inf)); worst = np.float32(np.inf)
            for j in range(N):
                c = cand[i, j]
                if c == gi or c < 0: continue
                ip = np.int32(0)
                for l in range(d): ip += np.int32(Xq[gi, l]) * np.int32(Xq[c, l])
                dist = np.float32(1.0) - np.float32(ip) * si * inv[c]
                if dist < worst:
                    p = K - 1
                    while p > 0 and dist < ds[p-1]:
                        ds[p] = ds[p-1]; ids[p] = ids[p-1]; p -= 1
                    ds[p] = dist; ids[p] = c; worst = ds[K-1]
            for j in range(K): knn_out[i, j] = ids[j]; d_out[i, j] = ds[j]

    @njit(fastmath=True, cache=True)
    def _fwht_row(x):
        # in-place FWHT on ONE row (length a power of two)
        d = x.shape[0]; h = 1
        while h < d:
            for i in range(0, d, 2 * h):
                for jj in range(i, i + h):
                    a = x[jj]; b = x[jj + h]
                    x[jj] = a + b
                    x[jj + h] = a - b
            h *= 2

    @njit(parallel=True, fastmath=True, cache=True)
    def _fwht_numba(X):
        # parallel over rows; the per-row transform lives in _fwht_row so the
        # prange body is a plain call. Inlining the nested variable-stride loops
        # directly under prange makes numba's parfor pass fail to build its CFG.
        for r in prange(X.shape[0]):
            _fwht_row(X[r])
except ImportError:
    _NUMBA = False
    _LUT = _kernel_u16 = _kernel_int8 = _fwht_row = _fwht_numba = None

# ======================================================================
# FWHT-JL projection (pads to next power of two)
# ======================================================================
def _next_pow2(d):
    p = 1
    while p < d: p <<= 1
    return p

def _fwht(x):
    """In-place FWHT on last axis; length must already be a power of two."""
    d = x.shape[-1]; nr = x.shape[0]; h = 1
    while h < d:
        v = x.reshape(nr, d // (2*h), 2, h)
        a = v[:, :, 0, :].copy(); b = v[:, :, 1, :].copy()
        v[:, :, 0, :] = a + b; v[:, :, 1, :] = a - b
        h *= 2
    return x

def _select_fwht():
    """Use the parallel numba kernel if it actually compiles on this numba build;
    otherwise fall back to the numpy transform (identical output). This keeps a
    numba/parfor incompatibility from ever crashing the run."""
    if _NUMBA and _fwht_numba is not None:
        try:
            _fwht_numba(np.ones((2, 4), dtype=np.float32))  # force compile now
            print("  FWHT: parallel numba kernel active")
            return _fwht_numba
        except Exception as e:
            print(f"  FWHT: numba kernel unavailable ({type(e).__name__}); using numpy fallback")
            return _fwht
    return _fwht
_FWHT = _select_fwht()

def _make_fjlt(D2, p, seed):
    rng = np.random.default_rng(seed)
    signs = (rng.integers(0, 2, D2, dtype=np.int8) * 2 - 1).astype(np.float32)
    coords = np.sort(rng.choice(D2, size=p, replace=False).astype(np.int64))
    return signs, coords

def load_and_project(h5_path, key, p, seed=SEED, chunk=50000):
    with h5py.File(h5_path, "r") as f:
        ds = f[key]; n, d = ds.shape
        D2 = _next_pow2(d)
        if p > D2: raise ValueError(f"proj p={p} exceeds padded dim {D2}")
        signs, coords = _make_fjlt(D2, p, seed)
        pad = D2 - d
        print(f"  FWHT-JL: d={d} -> pad {D2} -> subsample p={p}  (n={n})")
        Xp = np.empty((n, p), dtype=np.float32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                X = ds[i:j].astype(np.float32)
                nz = np.linalg.norm(X, axis=1, keepdims=True)
                bad = ~np.isfinite(nz.ravel()) | (nz.ravel() < 1e-20)
                if bad.any(): X[bad] = 0.0; nz[bad, 0] = 1.0
                np.divide(X, nz, out=X)
                if pad: X = np.ascontiguousarray(np.pad(X, ((0,0),(0,pad))))
                np.multiply(X, signs, out=X)
                _FWHT(X)
                sub = X[:, coords].copy()
                sn = np.linalg.norm(sub, axis=1, keepdims=True)
                bad = ~np.isfinite(sn.ravel()) | (sn.ravel() < 1e-20)
                if bad.any(): sub[bad] = 0.0; sn[bad, 0] = 1.0
                np.divide(sub, sn, out=sub)
                sub[~np.isfinite(sub)] = 0.0
                Xp[i:j] = sub
    return Xp, n, d

# ----------------------------------------------------------------------
# PCA projection (TruncatedSVD, no centering -> cosine-friendly).
# Fit components on a RANDOM sample (stratified contiguous blocks = fast
# sequential IO + representative even if the file is ordered), then transform
# all rows. Fit cost ~ sample size; transform ~ N*p. Renormalize after projecting.
# ----------------------------------------------------------------------
def read_sample(h5_path, key, sample_size, seed, chunk=100000):
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        ds = f[key]; n, d = ds.shape
        if sample_size >= n:                       # small set: just use all rows
            out = np.empty((n, d), dtype=np.float32)
            for i in range(0, n, chunk):
                j = min(i + chunk, n); out[i:j] = ds[i:j].astype(np.float32)
        else:                                      # stratified random blocks across the file
            n_strata = min(64, sample_size)
            per = max(1, sample_size // n_strata)
            stratum = n // n_strata
            blocks = []
            for s in range(n_strata):
                lo = s * stratum + int(rng.integers(0, max(1, stratum - per)))
                hi = min(lo + per, n)
                blocks.append(ds[lo:hi].astype(np.float32))
            out = np.concatenate(blocks, axis=0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        nz = np.linalg.norm(out, axis=1, keepdims=True)
        nz[~np.isfinite(nz) | (nz < 1e-20)] = 1.0
        out /= nz; out[~np.isfinite(out)] = 0.0
    return out

def pca_transform(h5_path, key, V, chunk=50000):
    """Project all rows onto PCA components V (p x d): L2-normalize in full dim,
    project, then renormalize for cosine in the reduced space. Streamed (memory-safe)."""
    Vt = np.ascontiguousarray(V.T); p = V.shape[0]
    with h5py.File(h5_path, "r") as f:
        ds = f[key]; n, d = ds.shape
        print(f"  PCA transform: d={d} -> p={p}  (n={n})")
        Xp = np.empty((n, p), dtype=np.float32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                X = ds[i:j].astype(np.float32)
                nz = np.linalg.norm(X, axis=1, keepdims=True)
                nz[~np.isfinite(nz) | (nz < 1e-20)] = 1.0
                X /= nz; X[~np.isfinite(X)] = 0.0
                Y = X @ Vt
                sn = np.linalg.norm(Y, axis=1, keepdims=True)
                sn[~np.isfinite(sn) | (sn < 1e-20)] = 1.0
                Y /= sn; Y[~np.isfinite(Y)] = 0.0
                Xp[i:j] = Y
    return Xp, n, d
def build_candidates(Xp, oversample, threads, delta=0.001):
    """Native build at this oversample. Returns int32 candidate matrix
    (n, oversample+1); PyNNDescent puts self at col 0 (we keep it; the kernel
    skips self, and self is re-prepended explicitly in the rerank)."""
    from pynndescent import NNDescent
    n = Xp.shape[0]
    # random_state MUST be None here: a fixed seed forces pynndescent's NN-descent
    # to run single-threaded (deterministic mode), ~4x slower at 8 threads. The graph
    # is approximate and feeds an exact rerank, so run-to-run jitter is harmless.
    nnd = NNDescent(Xp, n_neighbors=oversample + 1, metric="cosine",
                    n_trees=N_TREES, n_iters=N_ITERS, delta=delta, n_jobs=threads,
                    random_state=None, verbose=True)
    idx, _ = nnd.neighbor_graph
    return np.ascontiguousarray(idx.astype(np.int32))

def load_f16(h5_path, key, n, d, chunk=100000):
    Xf = np.empty((n, d), dtype=np.float16)
    with h5py.File(h5_path, "r") as f:
        ds = f[key]
        for i in range(0, n, chunk):
            j = min(i + chunk, n); Xf[i:j] = ds[i:j].astype(np.float16)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            c = Xf[i:j].astype(np.float32)
            nz = np.linalg.norm(c, axis=1, keepdims=True)
            bad = ~np.isfinite(nz.ravel()) | (nz.ravel() < 1e-20)
            if bad.any(): c[bad] = 0; nz[bad, 0] = 1
            np.divide(c, nz, out=c); c[~np.isfinite(c)] = 0
            Xf[i:j] = c.astype(np.float16)
    return Xf

def quantize_int8(Xf, chunk=100000):
    n, d = Xf.shape
    Xq = np.empty((n, d), dtype=np.int8); inv = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        c = Xf[i:j].astype(np.float32); c[~np.isfinite(c)] = 0
        am = np.abs(c).max(axis=1); am = np.where(am < 1e-12, 1.0, am)
        sc = (127.0 / am).astype(np.float32); inv[i:j] = (1.0 / sc).astype(np.float32)
        c *= sc[:, None]; np.clip(c, -127, 127, out=c); np.round(c, out=c)
        Xq[i:j] = c.astype(np.int8)
    return Xq, inv

def quantize_int8_from_disk(h5_path, key, n, d, chunk=100000):
    """Stream raw -> normalize -> int8 WITHOUT ever holding the full f16 array.
    Byte-identical to quantize_int8(load_f16(...)): the per-row f16 round-trip
    is replicated so the int8 codes (and thus recall) match the old path exactly.
    Peak here is just Xq (6.5 GB at full scale) + one chunk, not 13 GB + 6.5 GB."""
    Xq = np.empty((n, d), dtype=np.int8); inv = np.empty(n, dtype=np.float32)
    with h5py.File(h5_path, "r") as f:
        ds = f[key]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                c = ds[i:j].astype(np.float16).astype(np.float32)  # match load_f16: raw->f16 first
                nz = np.linalg.norm(c, axis=1, keepdims=True)
                bad = ~np.isfinite(nz.ravel()) | (nz.ravel() < 1e-20)
                if bad.any(): c[bad] = 0; nz[bad, 0] = 1
                np.divide(c, nz, out=c); c[~np.isfinite(c)] = 0
                c = c.astype(np.float16).astype(np.float32)   # match load_f16's f16 storage
                c[~np.isfinite(c)] = 0
                am = np.abs(c).max(axis=1); am = np.where(am < 1e-12, 1.0, am)
                sc = (127.0 / am).astype(np.float32); inv[i:j] = (1.0 / sc).astype(np.float32)
                c *= sc[:, None]; np.clip(c, -127, 127, out=c); np.round(c, out=c)
                Xq[i:j] = c.astype(np.int8)
    return Xq, inv

def rerank_topk_nonself(cand, k, quantize, Xq, inv, Xf, threads, chunk_rows=200000):
    """Return the k NEAREST NON-SELF neighbours: knns 1-based int32 (n,k), dists (n,k).
    Task 1 is a self-join and the challenge scores the k non-self neighbours, so self
    is excluded entirely -- keeping self in column 0 both wasted a neighbour slot and
    mismatched the ground truth (it cost ~0.03 recall on the eval)."""
    n = cand.shape[0]
    knn = np.empty((n, k), dtype=np.int32)
    dst = np.empty((n, k), dtype=np.float32)
    if quantize == "int8":
        _kernel_int8(Xq, inv, cand[:200], 0, k, knn[:200], dst[:200])   # warmup / JIT
        for i in range(0, n, chunk_rows):
            j = min(i + chunk_rows, n)
            _kernel_int8(Xq, inv, cand[i:j], i, k, knn[i:j], dst[i:j])
    else:
        Xu = Xf.view(np.uint16)
        _kernel_u16(Xu, cand[:200], _LUT, 0, k, knn[:200], dst[:200])    # warmup / JIT
        for i in range(0, n, chunk_rows):
            j = min(i + chunk_rows, n)
            _kernel_u16(Xu, cand[i:j], _LUT, i, k, knn[i:j], dst[i:j])
    knns = np.where(knn >= 0, knn + 1, 0)   # 1-based; 0 fills any slot short of k valid
    return knns, dst

def run_task1(input_path, key, k, output_dir, dataset, grid=None, threads=8,
              method="random", pca_sample=300000, pca_drop_top=0, delta=0.001):
    if grid is None:
        grid = GRID_TASK1_PCA if method == "pca" else GRID_TASK1
    print(f"Running task1 on {dataset}  ({len(grid)} configs, proj-method={method})")
    from collections import OrderedDict
    by_proj = OrderedDict()
    for c in grid: by_proj.setdefault(c["proj"], []).append(c)

    # PCA: fit components per distinct proj on ONE cached random sample, then free it
    # so the sample (~GB) is gone before the memory-heavy transform/build phase.
    Vs, fit_t = {}, {}
    if method == "pca":
        from sklearn.decomposition import TruncatedSVD
        sample = read_sample(input_path, key, pca_sample, SEED)
        print(f"  PCA fit on {sample.shape[0]} sampled rows (drop_top={pca_drop_top})")
        for proj in sorted({c["proj"] for c in grid}):
            t0 = time.time()
            svd = TruncatedSVD(n_components=proj + pca_drop_top, random_state=SEED).fit(sample)
            Vs[proj] = np.ascontiguousarray(
                svd.components_[pca_drop_top:pca_drop_top + proj].astype(np.float32))
            fit_t[proj] = time.time() - t0
            print(f"    fit {proj} comps in {fit_t[proj]:.1f}s")
        del sample; gc.collect()

    cand_dir = os.path.join(output_dir, "_cand"); os.makedirs(cand_dir, exist_ok=True)
    for proj, cfgs in by_proj.items():
        print(f"\n==== proj={proj} : {len(cfgs)} configs ====")
        t0 = time.time()
        if method == "pca":
            Xp, n, d = pca_transform(input_path, key, Vs[proj], chunk=50000)
            t_proj = (time.time() - t0) + fit_t[proj]      # honest end-to-end: fit + transform
        else:
            Xp, n, d = load_and_project(input_path, key, proj, chunk=50000)
            t_proj = time.time() - t0
        # ONE native build at the largest oversample; every smaller oversample is a
        # column slice of the same distance-sorted neighbour graph. (Building per
        # oversample, as before, repeats the expensive NN-descent once per config.)
        oss = sorted({c["oversample"] for c in cfgs})
        t0 = time.time()
        cand_full = build_candidates(Xp, oss[-1], threads, delta)   # (n, max_os+1), sorted near->far
        t_build = time.time() - t0
        del Xp; gc.collect()
        print(f"  built ONE graph at oversample={oss[-1]}, delta={delta} in {t_build:.1f}s "
              f"(sliced for oversamples {oss})")
        for N in oss:                                        # slice + disk-handoff per oversample
            np.save(os.path.join(cand_dir, f"cand_p{proj}_os{N}.npy"),
                    np.ascontiguousarray(cand_full[:, :N + 1]))
        del cand_full; gc.collect()
        build_t = {N: t_build for N in oss}   # one shared build; reported per config
        # rerank phase — never hold the f16 (13 GB) and int8 (6.5 GB) arrays at
        # once. int8 configs quantize straight from disk (no 13 GB buffer); the
        # f16 array is loaded only for "none" configs, after int8 is freed.
        # Peak ~16 GB (f16 path) / ~10 GB (int8 path) instead of ~23 GB.
        algo = "pca+pynndescent+rerank" if method == "pca" else ALGO
        pm = "pca" if method == "pca" else "fwht-jl"
        dt = f",droptop={pca_drop_top}" if (method == "pca" and pca_drop_top) else ""

        def _emit(c, Xq, inv, Xf):
            N, q = c["oversample"], c["quantize"]
            cand = np.load(os.path.join(cand_dir, f"cand_p{proj}_os{N}.npy"))
            t0 = time.time()
            knns, dists = rerank_topk_nonself(cand, k, q, Xq, inv, Xf, threads)
            t_rr = time.time() - t0; del cand; gc.collect()
            ident = f"index=({pm},proj={proj},os={N}{dt}),query=(rerank={q})"
            store_results(os.path.join(output_dir, f"{ident}.h5"), algo, dataset, "task1",
                          dists, knns, buildtime=t_proj + build_t[N], querytime=t_rr, params=ident)
            print(f"  wrote {ident}.h5  build={t_proj + build_t[N]:.1f}s query={t_rr:.1f}s")

        int8_cfgs = [c for c in cfgs if c["quantize"] == "int8"]
        none_cfgs = [c for c in cfgs if c["quantize"] != "int8"]

        if int8_cfgs:                                   # only Xq resident (~6.5 GB)
            Xq, inv = quantize_int8_from_disk(input_path, key, n, d)
            for c in int8_cfgs: _emit(c, Xq, inv, None)
            del Xq, inv; gc.collect()

        if none_cfgs:                                   # only Xf resident (~13 GB)
            Xf = load_f16(input_path, key, n, d)
            for c in none_cfgs: _emit(c, None, None, Xf)
            del Xf; gc.collect()

# ======================================================================
# Ceiling diagnostic: exact NN *in the projected space* on a query sample.
# Separates projection quality (this ceiling) from PyNNDescent's recovery
# (your eval pipeline recall). gap = ceiling - pipeline = PyNNDescent's loss.
# Same projection code as the pipeline (same PCA components / FWHT seed), so
# the two numbers are directly comparable per (proj, oversample).
# ======================================================================
def _project_all(input_path, key, proj, method, pca_sample, pca_drop_top):
    if method == "pca":
        from sklearn.decomposition import TruncatedSVD
        sample = read_sample(input_path, key, pca_sample, SEED)
        svd = TruncatedSVD(n_components=proj + pca_drop_top, random_state=SEED).fit(sample)
        V = np.ascontiguousarray(svd.components_[pca_drop_top:pca_drop_top + proj].astype(np.float32))
        del sample; gc.collect()
        return pca_transform(input_path, key, V)
    return load_and_project(input_path, key, proj)

def measure_ceiling(input_path, key, gt_key, k, output_dir, dataset, grid=None,
                    method="random", pca_sample=300000, pca_drop_top=0,
                    n_query=10000, seed=SEED):
    """For each (proj, oversample) in the grid: exact projected-space top-N pool on a
    query sample, then ceiling = fraction of the TRUE k-NN (from gt) the pool contains.
    Reported in the official self-including convention so it lines up with eval recall."""
    if grid is None:
        grid = GRID_TASK1_PCA if method == "pca" else GRID_TASK1
    pairs = sorted({(c["proj"], c["oversample"]) for c in grid})
    projs = sorted({p for p, _ in pairs})
    rng = np.random.default_rng(seed + 1)
    with h5py.File(input_path, "r") as f:
        n = f[key].shape[0]
        cur = f
        for kk in (gt_key if isinstance(gt_key, list) else gt_key.split("/")): cur = cur[kk]
        qidx = np.arange(n) if n_query >= n else np.sort(rng.choice(n, size=n_query, replace=False))
        gt_q = cur[qidx][:, :k].astype(np.int64)         # 1-based, self at col 0
    Q = len(qidx); true_ns = gt_q[:, 1:k] - 1            # (Q, k-1) 0-based true non-self
    print(f"Ceiling diagnostic on {dataset}: {Q} query pts, k={k}, gt='{gt_key}', method={method}")
    rows = []
    for proj in projs:
        Xp, nn, d = _project_all(input_path, key, proj, method, pca_sample, pca_drop_top)
        qblock = max(16, int(2e9 / (nn * 4)))            # cap the similarity block at ~2GB
        for (_, N) in [pr for pr in pairs if pr[0] == proj]:
            hit_off = 0; cap = 0
            for b in range(0, Q, qblock):
                qs = qidx[b:b + qblock]
                sims = Xp[qs] @ Xp.T                      # (qb, n) projected cosines
                for r in range(len(qs)):
                    sims[r, qs[r]] = -1e30               # exclude self -> N non-self
                    topN = np.argpartition(-sims[r], N)[:N]
                    c = int(np.intersect1d(topN, true_ns[b + r]).size)
                    hit_off += 1 + c; cap += c           # +1: self is a guaranteed hit
                del sims
            ceil = hit_off / (Q * k); capture = cap / (Q * (k - 1))
            ident = f"index=({'pca' if method=='pca' else 'fwht-jl'},proj={proj},os={N})"
            rows.append((proj, N, ceil, capture, ident))
            print(f"  proj={proj:4d} os={N:4d}:  ceiling@{k}={ceil:.4f}   non-self capture={capture:.4f}")
        del Xp; gc.collect()
    import csv
    path = os.path.join(output_dir, "ceiling.csv")
    with open(path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["dataset", "task", "proj", "oversample", "method",
                    "ceiling_recall", "nonself_capture", "n_query", "params"])
        for proj, N, ceil, capture, ident in rows:
            w.writerow([dataset, "task1", proj, N, method, f"{ceil:.6f}", f"{capture:.6f}", Q, ident])
    print(f"  wrote {path}  (compare ceiling_recall to your eval pipeline recall per proj/os)")

# ======================================================================
# Task 2 (Neyshabur-Srebro + brute force) and Task 3 (Gaussian-JL + sparse
# rerank): correct, compact. External queries -> no self issue; 1-based out.
# ======================================================================
def run_task2(data, queries, k, output_dir, dataset):
    print(f"Running task2 on {dataset}")
    db = np.asarray(data, np.float32); q = np.asarray(queries, np.float32)
    n, d = db.shape; nq = q.shape[0]
    t0 = time.time()
    nrm = np.linalg.norm(db[:100], axis=1)
    if not np.allclose(nrm, 1.0, atol=0.01):
        dn = np.linalg.norm(db, axis=1); M = float(dn.max()) * 1.001
        extra = np.sqrt(np.maximum(M*M - dn*dn, 0)).astype(np.float32)
        db = np.hstack([db, extra[:, None]]); db /= np.linalg.norm(db, axis=1, keepdims=True)
        q = np.hstack([q, np.zeros((nq, 1), np.float32)]); q /= np.linalg.norm(q, axis=1, keepdims=True)
    t_build = time.time() - t0
    t0 = time.time(); I = np.zeros((nq, k), np.int32); D = np.zeros((nq, k), np.float64)
    for i in range(0, nq, 500):
        j = min(i + 500, nq); sim = q[i:j] @ db.T
        for r in range(j - i):
            top = np.argpartition(-sim[r], k)[:k]; top = top[np.argsort(-sim[r, top])]
            I[i + r] = top + 1; D[i + r] = -sim[r, top]
    t_q = time.time() - t0
    ident = "neyshabur-srebro+bruteforce"
    store_results(os.path.join(output_dir, f"{ident}.h5"), ident, dataset, "task2",
                  D, I, t_build, t_q, ident)
    print(f"  wrote {ident}.h5  build={t_build:.1f}s query={t_q:.1f}s")

def run_task3(corpus, queries, k, output_dir, dataset, grid=((256, 10000), (256, 7500))):
    print(f"Running task3 on {dataset}")
    from scipy import sparse
    V = max(corpus.shape[1], queries.shape[1])
    if corpus.shape[1] < V: corpus = sparse.hstack([corpus, sparse.csr_matrix((corpus.shape[0], V-corpus.shape[1]), dtype=np.float32)]).tocsr()
    if queries.shape[1] < V: queries = sparse.hstack([queries, sparse.csr_matrix((queries.shape[0], V-queries.shape[1]), dtype=np.float32)]).tocsr()
    nq = queries.shape[0]
    for proj, cand_n in grid:
        np.random.seed(SEED)
        R = (np.random.randn(V, proj) / np.sqrt(proj)).astype(np.float32)
        t0 = time.time()
        dbp = np.asarray(corpus @ R, dtype=np.float32); qp = np.asarray(queries @ R, dtype=np.float32)
        t_build = time.time() - t0
        t0 = time.time(); I = np.zeros((nq, k), np.int32); D = np.zeros((nq, k), np.float64)
        for i in range(0, nq, 100):
            j = min(i + 100, nq); sims = qp[i:j] @ dbp.T
            for r in range(j - i):
                m = min(cand_n, corpus.shape[0])
                c = np.argpartition(-sims[r], m-1)[:m]
                ex = np.asarray((corpus[c] @ queries[i+r].T).todense()).ravel()
                top = np.argsort(-ex)[:k]
                I[i + r] = c[top] + 1; D[i + r] = -ex[top]
        t_q = time.time() - t0
        ident = f"index=(gauss-jl,proj={proj}),query=(cand={cand_n})"
        store_results(os.path.join(output_dir, f"{ident}.h5"), "gauss-jl+sparse-rerank",
                      dataset, "task3", D, I, t_build, t_q, ident)
        print(f"  wrote {ident}.h5  build={t_build:.1f}s query={t_q:.1f}s")

# ======================================================================
# Main — the TIRA contract
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--task-description", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 8)))
    ap.add_argument("--proj-method", choices=["random", "pca"], default="random",
                    help="random = FWHT-JL (default); pca = TruncatedSVD fit-on-sample")
    ap.add_argument("--pca-sample", type=int, default=300000,
                    help="rows sampled to FIT PCA components (>= n reads all rows)")
    ap.add_argument("--pca-drop-top", type=int, default=0,
                    help="discard the top-k PCA components before projecting")
    ap.add_argument("--measure-ceiling", action="store_true",
                    help="task1 diagnostic: exact projected-space NN on a query sample "
                         "(the projection-quality ceiling) instead of the full pipeline")
    ap.add_argument("--ceiling-queries", type=int, default=10000,
                    help="number of query points sampled for --measure-ceiling")
    ap.add_argument("--delta", type=float, default=0.01,
                    help="task1 PyNNDescent early-stop threshold; larger stops the descent "
                         "sooner. The p=320 sweep showed higher delta gives no build saving "
                         "there and slightly lower recall, so 0.01 is the tuned value.")
    a = ap.parse_args()

    cfg = load_task_config(a.task_description)
    task = cfg["task"]; k = cfg.get("k", 15)
    dataset = cfg["dataset_name"]; key = cfg["data"]
    os.makedirs(a.output, exist_ok=True)
    print(f"task={task} k={k} dataset={dataset} input={a.input}")

    if task == "task1":
        if a.measure_ceiling:
            # random (FWHT) ceiling sweeps projection dimension; PCA keeps its own grid
            cgrid = None if a.proj_method == "pca" else GRID_TASK1_CEILING
            measure_ceiling(a.input, key, cfg.get("gt_I", ["allknn", "knns"]), k, a.output, dataset,
                            grid=cgrid, method=a.proj_method, pca_sample=a.pca_sample,
                            pca_drop_top=a.pca_drop_top, n_query=a.ceiling_queries)
        else:
            # stream from the path (memory-safe); never eager-load the full f32 array
            run_task1(a.input, key, k, a.output, dataset, threads=a.threads,
                      method=a.proj_method, pca_sample=a.pca_sample, pca_drop_top=a.pca_drop_top,
                      delta=a.delta)
    elif task == "task2":
        with h5py.File(a.input, "r") as f:
            data = f[key][()]
            q = f
            qk = cfg["queries"]; cur = f
            for p in (qk if isinstance(qk, list) else qk.split("/")): cur = cur[p]
            queries = cur[()]
        run_task2(data, queries, k, a.output, dataset)
    elif task == "task3":
        from scipy.sparse import csr_matrix
        def load_sp(g):
            return csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                              shape=tuple(g.attrs["shape"]))
        with h5py.File(a.input, "r") as f:
            dk = key; cur = f
            for p in (dk if isinstance(dk, list) else dk.split("/")): cur = cur[p]
            corpus = load_sp(cur)
            qk = cfg["queries"]; cur = f
            for p in (qk if isinstance(qk, list) else qk.split("/")): cur = cur[p]
            queries = load_sp(cur)
        run_task3(corpus, queries, k, a.output, dataset)
    else:
        print(f"Unknown task {task}"); sys.exit(1)

if __name__ == "__main__":
    main()
