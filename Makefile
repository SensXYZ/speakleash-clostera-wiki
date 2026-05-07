.PHONY: help setup sync lint format typecheck check clean \
       sample embed cluster label visualize pipeline

# ── Configuration ────────────────────────────────────────────
COUNT       ?= 10k
SEED        ?= 42
MODEL_EMBED ?= text-embedding-bge-m3
MODEL_INSTR ?= bielik-11b-v3.0-instruct
API_KEY     ?= $(LMSTUDIO_API_KEY)
K           ?= 64
REDUCER     ?= umap

# ── Help ─────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────
setup: ## Install all dependencies (uv sync)
	uv sync

sync: setup  ## Alias for setup

# ── Pipeline ─────────────────────────────────────────────────
sample: ## Step 1: Sample articles from Polish Wikipedia
	uv run 1_create_dataset.py -n $(COUNT) --seed $(SEED)

embed: ## Step 2: Embed + cluster (needs local model server)
	uv run 2_embed_and_cluster.py --model $(MODEL_EMBED) --api-key "$(API_KEY)" -k $(K)

cluster: ## Step 2 (skip embed): Re-cluster with different k/metric
	uv run 2_embed_and_cluster.py --skip-embed --model unused -k $(K)

label: ## Step 3: Label clusters + visualize
	uv run 3_label_and_visualize.py --model $(MODEL_INSTR) --api-key "$(API_KEY)"

visualize: ## Step 3 (skip labeling): Re-render plot only
	uv run 3_label_and_visualize.py --skip-labeling --3d

pipeline: sample embed label ## Run full pipeline (1 → 2 → 3)
	@echo "Done. Open out/clusters_2d.html in your browser."

# ── Linting & Quality ────────────────────────────────────────
lint: ## Run ruff linter
	uv run ruff check .

format: ## Run ruff formatter
	uv run ruff format .

typecheck: ## Run mypy type checker
	uv run mypy *.py

check: lint typecheck ## Run all checks (lint + typecheck)

# ── Cleanup ──────────────────────────────────────────────────
clean: ## Remove generated files (out/, wiki_pl.jsonl, checkpoint)
	rm -rf out/
	rm -f wiki_pl.jsonl
