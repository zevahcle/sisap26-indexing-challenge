# SISAP 2026 Indexing Challenge — team `wh-team` (`late-shark`)

Companion code for *"Three Regimes, One Architecture: Walsh–Hadamard
Projections and Exact Rerank in the SISAP 2026 Indexing Challenge."*

One architecture across all three tasks: **project to a cheap
representation → generate candidates → rerank exactly in the original
space.** The angular gap of each task's nearest neighbours selects the
regime.

| Task | Data | Method | dev recall | dev time |
|------|------|--------|-----------|----------|
| 1 | Wikipedia BGE-M3, 1024-d cosine, 6.35M, self-join | FWHT–JL → 256 + PyNNDescent + int8 rerank | 0.8223 | 12.3 min |
| 2 | Llama 128-d MIPS, 256K | Neyshabur–Srebro augmentation + brute force | 1.0000 | 5.2 s |
| 3 | NQ SPLADE-v3 sparse dot, 2.68M | Gaussian-JL densification → 256 + exact sparse rerank | 0.9250 | 7.0 min |

`search.py` is the complete submission — all three task pipelines are
inline. Task identity, `k`, and HDF5 keys come from `config.json`, per the
TIRA contract:

```
search.py --input <benchmark.h5> --task-description <config.json> --output <dir>
```

## Files

- `search.py` — entry point (all tasks)
- `Dockerfile`, `requirements.txt`, `.dockerignore` — container build
- Dependencies are pure Python (numpy, scipy, h5py, numba, pynndescent) and
  install as manylinux wheels; numba JIT-compiles at runtime on the target
  CPU. No compiler, no `-march=native`, nothing bound to the build machine.

## Resource budget

The eval container is **24 GB RAM / 8 CPU**. Task 1 at 6.35M × 1024 is the
tight one: the pipeline streams the projection from disk, hands candidate
matrices off to disk between the graph build and rerank, quantizes to int8
by streaming (never materialising the 13 GB f16 array), and reranks the
int8 configs and the f16 configs in separate phases so the two large arrays
are never resident together. Peak stays ≈16 GB.

## Build and test locally under the eval cap

The spot-check datasets are too small to exercise the memory ceiling — test
against **wikipedia-dev** with the cap imposed, or Task 1 may pass locally
and be OOM-killed on the eval node.

```bash
docker build -t wh-team .

# wikipedia-dev: https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/tree/main/wikipedia
docker run --rm --memory=24g --memory-swap=24g --cpus=8 \
  -v /path/to/wikipedia-dev:/data:ro -v /tmp/out:/out \
  wh-team \
  /app/search.py --input /data/<benchmark>.h5 \
                 --task-description /data/config.json \
                 --output /out
```

`--memory-swap` equal to `--memory` disables swap, matching the eval node so
an over-budget run fails fast instead of paging. A successful Task 1 run
writes 15 `.h5` files (the config ladder) to `/out`.

## Submit to TIRA

```bash
export BUILDX_NO_DEFAULT_ATTESTATIONS=1   # single-manifest image for the cluster

tira-cli code-submission --path . \
  --command '/app/search.py --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir' \
  --task sisap-2026 --dataset task-1-spot-check-20260602-training --dry-run
```

Drop `--dry-run` for the real submission. Requires a git repo with a
committed, clean tree and a remote.
