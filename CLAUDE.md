# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Three-step pipeline for sampling Polish Wikipedia articles (via `speakleash`), embedding them with a local model server, clustering with `clostera`, and labeling/visualizing the results. All scripts are standalone CLI tools with `argparse` — no shared modules, no package structure.

Each script has **PEP 723 inline metadata** declaring its dependencies, so `uv run <script.py>` works without a pre-created venv.

## Requirements

- **Python 3.10–3.13**
- On macOS Apple Silicon, use native arm64 Python (clostera has arm64 wheels; x86_64/Rosetta requires source compilation).
- A local model server with OpenAI-compatible API (`/v1/embeddings` and `/v1/chat/completions`):
  - **LM Studio** (GUI) — Developer tab, default `http://localhost:1234/v1`
  - **llama.cpp server** (CLI) — `llama-server -m model.gguf --port 1234`, same endpoint
- If authentication is on, store the key in `.env` as `LMSTUDIO_API_KEY=sk-lm-...` (`.env` is gitignored). Scripts auto-load `.env` via `python-dotenv` — no manual `source` needed. A template is in `.env.example`.

## Setup

```bash
# Option A: uv run (no manual venv — uses PEP 723 inline metadata)
uv run 1_create_dataset.py --help

# Option B: install all deps upfront via pyproject.toml
uv sync

# Option C: classic pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

Scripts must run in numbered order. Each reads output from the previous step.

```bash
# Step 1: Sample articles -> wiki_pl.jsonl
uv run 1_create_dataset.py -n 10k --seed 42

# Step 2: Embed + cluster -> out/vectors.npy, out/labels.npy, out/wiki_pl_clustered.jsonl, out/clusters/
uv run 2_embed_and_cluster.py --model text-embedding-bge-m3 --api-key "$LMSTUDIO_API_KEY" -k 128

# Step 3: Label clusters via LLM + UMAP/plotly visualization -> out/cluster_labels.json, out/clusters_2d.html
uv run 3_label_and_visualize.py --model bielik-11b-v3.0-instruct --api-key "$LMSTUDIO_API_KEY"

# Re-render plot without re-querying LLM:
uv run 3_label_and_visualize.py --skip-labeling --3d
```

### Using llama.cpp server instead of LM Studio

```bash
# Start embedding model
llama-server -m bge-m3.gguf --port 1234 --embedding

# Run step 2 (model name = GGUF filename or alias)
uv run 2_embed_and_cluster.py --model bge-m3.gguf --api-key unused -k 128

# Then switch to instruct model for step 3
llama-server -m bielik-11b-v3.0-instruct-Q4_K_M.gguf --port 1234
uv run 3_label_and_visualize.py --model bielik-11b-v3.0-instruct-Q4_K_M.gguf --api-key unused
```

## Architecture

### Data Flow

```
speakleash (plwiki) ──> 1_create_dataset.py ──> wiki_pl.jsonl
                                                    │
                    local server (embeddings) ──> 2_embed_and_cluster.py ──> out/
                                                    │                        vectors.npy
                                                    │                        labels.npy
                                                    │                        *_clustered.jsonl
                                                    │                        clusters/*.jsonl
                                                    │
                    local server (instruct)   ──> 3_label_and_visualize.py ──> cluster_labels.json
                                                                                  clusters_2d/3d.html
```

### Script Conventions

- All scripts use `if __name__ == "__main__": sys.exit(main())` with `main() -> int` returning exit codes
- PEP 723 `# /// script` blocks declare per-script dependencies — `uv run` resolves them automatically
- Progress and diagnostics go to stderr; data goes to stdout/files
- JSONL format: one JSON object per line, each with `"text"` and `"meta"` keys; step 2 adds `"cluster"`
- `clostera.Clusterer` is used for clustering (wraps GPU/CPU backends automatically based on hardware)
- The `openai` Python SDK is the HTTP client for the local server's OpenAI-compatible API (not a cloud OpenAI service)
- Scripts 2 & 3 auto-load `.env` via `python-dotenv` (`load_dotenv(override=False)`) — `--api-key` defaults to `$LMSTUDIO_API_KEY`
- Script 2 uses embedding checkpoint/resume (`vectors_partial.npy` + `.meta.json` in `out/`) — interrupted runs pick up where they left off if config matches
- OpenAI client uses `max_retries` (5 for embedding, 3 for labeling) for automatic retry/backoff on transient failures

### Key Output Files (in `out/`)

| File | Description |
|------|-------------|
| `vectors.npy` | Float32 embedding matrix `(N, D)`, row-aligned with JSONL |
| `labels.npy` | Integer cluster labels, one per article |
| `*_clustered.jsonl` | Input JSONL with added `"cluster"` field |
| `clusters/cluster_NNN.jsonl` | Per-cluster JSONL splits (zero-padded) |
| `cluster_labels.json` | Map of `cluster_id → Polish name` from LLM |
| `clusters_2d.html` / `clusters_3d.html` | Interactive Plotly scatter (UMAP or PCA) |

### Useful CLI Flags for Iteration

- `--skip-embed` (step 2): re-cluster with different `-k`/`--metric` without re-embedding
- `--skip-labeling` (step 3): re-render visualization without re-querying LLM
- `--limit` (step 2): cap article count for debugging
- `--reducer pca` (step 3): use PCA instead of UMAP

## Linting, Type Checking & Tests

```bash
uv run ruff check .            # lint
uv run ruff check --fix .      # lint with auto-fix
uv run ruff format .           # format
uv run mypy *.py               # type check
uv run pytest tests/ -v        # run tests
```

Or use the Makefile:

```bash
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy
make test        # pytest
make check       # lint + typecheck + test
```

```bash
uv run ruff check .            # lint
uv run ruff check --fix .      # lint with auto-fix
uv run ruff format .           # format
uv run mypy *.py               # type check
```

Or use the Makefile:

```bash
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy
make check       # lint + typecheck
```

## Dependencies

| Package | Role |
|---------|------|
| `speakleash` | Streams Polish corpora (plwiki dataset) |
| `clostera` | Clustering (GPU-accelerated when available, CPU fallback) |
| `openai` | HTTP client for `/v1/embeddings` and `/v1/chat/completions` on local server |
| `numpy` | Vector storage (`.npy` files), cosine similarity in step 3 |
| `scikit-learn` | PCA fallback when `umap-learn` is not installed |
| `umap-learn` | Dimensionality reduction for visualization |
| `plotly` | Interactive HTML scatter plots |
| `python-dotenv` | Auto-loads `.env` for `LMSTUDIO_API_KEY` |
