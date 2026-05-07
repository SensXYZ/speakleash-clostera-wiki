# speakleash-clostera-wiki

**English** | [Polski](README.md)

Scripts for building datasets from Polish Wikipedia using the [`speakleash`](https://pypi.org/project/speakleash/) package.

## Requirements

- **Python 3.10–3.13**
- On macOS Apple Silicon, use **native arm64 Python** — `clostera` has pre-built arm64 wheels; x86_64 (Rosetta) requires source compilation.
- A local model server with an OpenAI-compatible API:
  - **LM Studio** (GUI) — **Developer / Local Server** tab
  - **llama.cpp server** (CLI) — see section below
- Dependencies from [requirements.txt](requirements.txt) or [pyproject.toml](pyproject.toml)

## Installation

### Option A: `uv run` — no manual venv (recommended)

Scripts include PEP 723 inline metadata — `uv` automatically creates an environment and installs dependencies on first run:

```bash
uv run 1_create_dataset.py --help
```

To install everything upfront:

```bash
uv sync
```

### Option B: classic venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model Server

Scripts communicate with a local server via the OpenAI-compatible API (`/v1/embeddings` and `/v1/chat/completions`). You can use LM Studio or llama.cpp.

### LM Studio (GUI)

1. Open LM Studio, go to **Developer / Local Server** and start the server (default: `http://localhost:1234`).
2. Load an embedding model (e.g. `text-embedding-bge-m3`) and/or an instruct model (e.g. `bielik-11b-v3.0-instruct`).
3. If you have authentication enabled (**Developer → Authentication**), save the token in `.env`:

```bash
echo 'LMSTUDIO_API_KEY=sk-lm-...' > .env
set -a && source .env && set +a
```

### llama.cpp server (CLI)

A GUI-free alternative — full control from the terminal. Install via Homebrew:

```bash
brew install llama.cpp
```

Start the server:

```bash
# Embedding model (step 2)
llama-server \
  -m ~/.cache/lm-studio/models/bge-m3.gguf \
  --port 1234 \
  --embedding

# Instruct model (step 3)
llama-server \
  -m ~/.cache/lm-studio/models/bielik-11b-v3.0-instruct-Q4_K_M.gguf \
  --port 1234 \
  --host 127.0.0.1
```

The `--model` flag in the scripts corresponds to the GGUF filename (or alias). Example:

```bash
uv run 2_embed_and_cluster.py \
  --model bge-m3.gguf \
  --api-key unused \
  -k 128

uv run 3_label_and_visualize.py \
  --model bielik-11b-v3.0-instruct-Q4_K_M.gguf \
  --api-key unused
```

> **Note:** llama.cpp server does not verify the API key — `--api-key unused` is sufficient. The default endpoint `http://localhost:1234/v1` is the same as LM Studio, so no `--base-url` change is needed.

## Pipeline — execution order

```bash
# 1) Sample articles from Polish Wikipedia (output: wiki_pl.jsonl)
uv run 1_create_dataset.py -n 10k --seed 42

# 2) Embedding + clustering
#    output: out/vectors.npy, out/labels.npy,
#            out/wiki_pl_clustered.jsonl, out/clusters/cluster_NNN.jsonl
uv run 2_embed_and_cluster.py \
  --model text-embedding-bge-m3 \
  --api-key "$LMSTUDIO_API_KEY" \
  -k 128

# 3) Label clusters via LLM + UMAP/plotly visualization
#    output: out/cluster_labels.json, out/clusters_2d.html / clusters_3d.html
uv run 3_label_and_visualize.py \
  --model bielik-11b-v3.0-instruct \
  --api-key "$LMSTUDIO_API_KEY"

# 3a) Quick plot iteration without re-querying the LLM
uv run 3_label_and_visualize.py --skip-labeling --3d
open out/clusters_3d.html
```

## Scripts

### `1_create_dataset.py` — Random sample from Polish Wikipedia

Downloads the `plwiki` dataset via `speakleash`, randomly samples the requested number of articles, and writes them to a JSONL file (one article per line).

**Output format** (each line is a separate JSON object):

```json
{"text": "article content...", "meta": {"title": "...", "length": 1234, "sentences": 12, "words": 200, ...}}
```

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| `-n`, `--count` | `10k` | number of articles to sample (accepts `10000`, `10k`, `100K`, `1M`) |
| `-o`, `--output` | `wiki_pl.jsonl` | output JSONL path |
| `--dataset` | `plwiki` | speakleash dataset name |
| `--cache-dir` | `datasets` | local cache directory for speakleash downloads |
| `--seed` | none | RNG seed for reproducible sampling |

**Examples:**

```bash
# 10,000 articles -> wiki_pl.jsonl
uv run 1_create_dataset.py -n 10k

# 100,000 articles, deterministically
uv run 1_create_dataset.py -n 100k --seed 42

# Custom output path
uv run 1_create_dataset.py -n 50000 -o data/wiki_pl_50k.jsonl
```

**How sampling works:** the script reads the document count from the dataset (`ds.documents`), draws `N` unique indices via `random.sample`, then makes a single pass over the `ds.ext_data` stream and writes only the selected entries. The full dataset still gets downloaded to `--cache-dir` — this is inherent to how `speakleash` works.

### `2_embed_and_cluster.py` — Embedding + clostera clustering

Reads the JSONL produced by script #1, calls the local embedding endpoint (OpenAI-compatible `POST /v1/embeddings`) for each article, saves the `float32` vector matrix as `vectors.npy`, then clusters them with [`clostera.Clusterer`](https://pypi.org/project/clostera/) and saves labels and an enriched JSONL.

**Server requirements:**

1. Start a server (LM Studio or llama.cpp) on `http://localhost:1234`.
2. Load an embedding model (e.g. `text-embedding-bge-m3`, `text-embedding-multilingual-e5-large`). The model identifier you enter here is passed via `--model`.

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | `wiki_pl.jsonl` | input JSONL |
| `-o`, `--output-dir` | `out` | directory for `vectors.npy`, `labels.npy` and enriched JSONL |
| `--model` | (required) | embedding model id loaded in the server |
| `--base-url` | `http://localhost:1234/v1` | server address |
| `--api-key` | `lm-studio` | any non-empty string |
| `--batch-size` | `32` | texts per request |
| `--max-chars` | `2000` | truncate each article before sending (most models have a ~512 token limit) |
| `-k`, `--clusters` | `64` | number of clusters |
| `--metric` | `cos` | `cos` or `l2` |
| `--algorithm` | `auto` | clostera algorithm name (see `clostera.available_algorithms()`) |
| `--limit` | none | cap number of articles (debug) |
| `--skip-embed` | off | skip embedding and load `vectors.npy` from `--output-dir` (for clustering tuning) |

**Examples:**

```bash
# Full pipeline: embed + cluster, k=64, metric=cos
uv run 2_embed_and_cluster.py \
  -i wiki_pl.jsonl \
  --model text-embedding-bge-m3 \
  -k 64

# Experiment with cluster count without re-embedding
uv run 2_embed_and_cluster.py --skip-embed -k 256 --metric cos --model unused
```

**Output files** (in `--output-dir`):

- `vectors.npy` — `float32` of shape `(N, D)`, one row per article, in input JSONL order
- `labels.npy` — cluster labels (one per article)
- `wiki_pl_clustered.jsonl` — copy of input with added `"cluster": <int>` field per line
- `clusters/cluster_000.jsonl` … `clusters/cluster_NNN.jsonl` — JSONL split per cluster (zero-padded numbers; the directory is cleaned on each run so changing `-k` doesn't leave stale files)

**Inspect cluster contents:**

```bash
# Titles from cluster 5
jq -r '.meta.title' out/clusters/cluster_005.jsonl

# Cluster sizes, descending
wc -l out/clusters/cluster_*.jsonl | sort -rn | head -20
```

### `3_label_and_visualize.py` — LLM labeling + UMAP/plotly visualization

For each cluster, selects N articles closest to the centroid (cosine similarity), sends them to an instruct model requesting a short Polish cluster name, reduces vectors to 2D/3D via UMAP, and renders an interactive plotly HTML chart.

**Server requirements:** a Polish-capable instruct model (e.g. `bielik-11b-v3.0-instruct`, `bielik-4.5b-v3.0-instruct@q8_0`).

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input-dir` | `out` | directory with `vectors.npy`, `labels.npy`, `*_clustered.jsonl` |
| `-o`, `--output-dir` | `=input-dir` | output directory for `cluster_labels.json` and HTML |
| `--clustered-jsonl` | (auto) | specific path to clustered JSONL; defaults to first `*_clustered.jsonl` |
| `--model` | required (unless `--skip-labeling`) | model id in the server |
| `--base-url` | `http://localhost:1234/v1` | server endpoint |
| `--api-key` | `lm-studio` | server token |
| `--samples-per-cluster` | `8` | articles per cluster sent to the LLM |
| `--max-chars` | `400` | truncate each sample before the prompt |
| `--temperature` | `0.2` | model temperature |
| `--reducer` | `umap` | `umap` or `pca` |
| `--3d` | off | render 3D scatter instead of 2D |
| `--annotate-top` | `30` | how many largest clusters get text labels at centroid |
| `--plot-name` | `clusters_2d.html`/`clusters_3d.html` | output HTML filename |
| `--skip-labeling` | off | skip LLM calls and reuse existing `cluster_labels.json` |

**Examples:**

```bash
# Full run: label clusters via Bielik and generate 2D plot
uv run 3_label_and_visualize.py \
  --model bielik-11b-v3.0-instruct \
  --api-key "$LMSTUDIO_API_KEY"

# 3D, without re-querying the LLM
uv run 3_label_and_visualize.py --skip-labeling --3d

# Tune UMAP/PCA, separate output file
uv run 3_label_and_visualize.py --skip-labeling --reducer pca --plot-name clusters_pca.html
```

**Output files:**

- `cluster_labels.json` — map of `cluster_id → name` (saved after LLM calls, read when `--skip-labeling`)
- `clusters_2d.html` or `clusters_3d.html` — interactive plotly chart. Hover on a point = cluster name + article title. Click legend = hide/show cluster, double-click = solo.

**Open in browser** (VS Code preview renders plotly poorly):

```bash
open out/clusters_2d.html        # or clusters_3d.html
```
