# speakleash-clostera-wiki

[English](README.en.md) | **Polski**

Skrypty do budowy zbiorów danych z polskiej Wikipedii w oparciu o pakiet [`speakleash`](https://pypi.org/project/speakleash/).

## Wymagania

- **Python 3.10–3.13**
- Na macOS Apple Silicon najlepiej użyć **natywnego arm64 Pythona** — `clostera` ma gotowe wheele dla arm64, dla x86_64 (Rosetta) trzeba by kompilować ze źródeł.
- Serwer modeli lokalnych z OpenAI‑kompatybilnym API:
  - **LM Studio** (GUI) — zakładka **Developer / Local Server**
  - **llama.cpp server** (CLI) — patrz sekcja poniżej
- Zależności z [requirements.txt](requirements.txt) lub [pyproject.toml](pyproject.toml)

## Instalacja

### Opcja A: `uv run` — bez ręcznego venv (zalecane)

Skrypty mają wbudowane metadane PEP 723 — `uv` automatycznie utworzy środowisko i zainstaluje zależności przy pierwszym uruchomieniu:

```bash
uv run 1_create_dataset.py --help
```

Jeśli chcesz zainstalować wszystko z góry:

```bash
uv sync
```

### Opcja B: klasyczny venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Serwer modeli

Skrypty komunikują się z lokalnym serwerem przez OpenAI‑kompatybilne API (`/v1/embeddings` i `/v1/chat/completions`). Możesz użyć LM Studio lub llama.cpp.

### LM Studio (GUI)

1. Otwórz LM Studio, w zakładce **Developer / Local Server** wystartuj serwer (domyślnie `http://localhost:1234`).
2. Wczytaj model embeddingowy (np. `text-embedding-bge-m3`) i/lub model instrukcyjny (np. `bielik-11b-v3.0-instruct`).
3. Jeśli masz włączoną autoryzację (`Developer → Authentication`), zapisz token w `.env`:

```bash
echo 'LMSTUDIO_API_KEY=sk-lm-...' > .env
set -a && source .env && set +a
```

### llama.cpp server (CLI)

Alternatywa bez GUI — pełna kontrola z terminala. Instalacja przez Homebrew:

```bash
brew install llama.cpp
```

Uruchomienie serwera:

```bash
# Model embeddingowy (krok 2)
llama-server \
  -m ~/.cache/lm-studio/models/bge-m3.gguf \
  --port 1234 \
  --embedding

# Model instrukcyjny (krok 3)
llama-server \
  -m ~/.cache/lm-studio/models/bielik-11b-v3.0-instruct-Q4_K_M.gguf \
  --port 1234 \
  --host 127.0.0.1
```

Flagi `--model` w skryptach odpowiadają nazwie pliku GGUF (lub aliasowi). Przykład:

```bash
uv run 2_embed_and_cluster.py \
  --model bge-m3.gguf \
  --api-key unused \
  -k 128

uv run 3_label_and_visualize.py \
  --model bielik-11b-v3.0-instruct-Q4_K_M.gguf \
  --api-key unused
```

> **Uwaga:** llama.cpp server nie weryfikuje klucza API — `--api-key unused` wystarczy. Domyślny endpoint `http://localhost:1234/v1` jest ten sam co w LM Studio, więc nie trzeba zmieniać `--base-url`.

## Pipeline — kolejność uruchamiania

```bash
# 1) wylosuj próbkę z polskiej Wikipedii (output: wiki_pl.jsonl)
uv run 1_create_dataset.py -n 10k --seed 42

# 2) embedding + klastrowanie
#    output: out/vectors.npy, out/labels.npy,
#            out/wiki_pl_clustered.jsonl, out/clusters/cluster_NNN.jsonl
uv run 2_embed_and_cluster.py \
  --model text-embedding-bge-m3 \
  --api-key "$LMSTUDIO_API_KEY" \
  -k 128

# 3) etykietowanie klastrów przez LLM + wizualizacja UMAP/plotly
#    output: out/cluster_labels.json, out/clusters_2d.html / clusters_3d.html
uv run 3_label_and_visualize.py \
  --model bielik-11b-v3.0-instruct \
  --api-key "$LMSTUDIO_API_KEY"

# 3a) szybkie iteracje na wykresie bez ponownego pytania LLM
uv run 3_label_and_visualize.py --skip-labeling --3d
open out/clusters_3d.html
```

## Skrypty

### `1_create_dataset.py` — losowy próbka z polskiej Wikipedii

Pobiera dataset `plwiki` przez `speakleash`, losuje zadaną liczbę artykułów i zapisuje je do pliku JSONL (jeden artykuł na linię).

**Format wyjścia** (każda linia to osobny JSON):

```json
{"text": "treść artykułu...", "meta": {"title": "...", "length": 1234, "sentences": 12, "words": 200, ...}}
```

**Argumenty CLI:**

| Flaga | Domyślnie | Opis |
|---|---|---|
| `-n`, `--count` | `10k` | liczba artykułów do wylosowania (akceptuje `10000`, `10k`, `100K`, `1M`) |
| `-o`, `--output` | `wiki_pl.jsonl` | ścieżka pliku wyjściowego |
| `--dataset` | `plwiki` | nazwa datasetu speakleash |
| `--cache-dir` | `datasets` | katalog cache, do którego speakleash ściąga dane |
| `--seed` | brak | seed RNG do powtarzalnego losowania |

**Przykłady:**

```bash
# 10 000 artykułów -> wiki_pl.jsonl
uv run 1_create_dataset.py -n 10k

# 100 000 artykułów, deterministycznie
uv run 1_create_dataset.py -n 100k --seed 42

# inna ścieżka wyjścia
uv run 1_create_dataset.py -n 50000 -o data/wiki_pl_50k.jsonl
```

**Jak działa losowanie:** skrypt odczytuje liczbę dokumentów w datasecie (`ds.documents`), losuje `N` unikalnych indeksów przez `random.sample`, a następnie jednokrotnie iteruje strumień `ds.ext_data` i zapisuje tylko wybrane pozycje. Pełny dataset i tak musi się pobrać do `--cache-dir` — to wynika ze sposobu działania `speakleash`.

### `2_embed_and_cluster.py` — embeddingi + klastrowanie clostera

Czyta JSONL wyprodukowany przez skrypt #1, dla każdego artykułu woła lokalny endpoint embeddingowy (OpenAI‑kompatybilny `POST /v1/embeddings`), zapisuje macierz wektorów `float32` jako `vectors.npy`, a następnie klastruje je przez [`clostera.Clusterer`](https://pypi.org/project/clostera/) i zapisuje etykiety oraz wzbogacony JSONL.

**Wymagania po stronie serwera:**

1. Wystartuj serwer (LM Studio lub llama.cpp) na `http://localhost:1234`.
2. Wczytaj model embeddingowy (np. `text-embedding-bge-m3`, `text-embedding-multilingual-e5-large`). Identyfikator modelu, który tu wpiszesz, podajesz dalej w `--model`.

**Argumenty CLI:**

| Flaga | Domyślnie | Opis |
|---|---|---|
| `-i`, `--input` | `wiki_pl.jsonl` | wejściowy JSONL |
| `-o`, `--output-dir` | `out` | katalog na `vectors.npy`, `labels.npy` i wzbogacony JSONL |
| `--model` | (wymagane) | id modelu embeddingowego załadowanego w serwerze |
| `--base-url` | `http://localhost:1234/v1` | adres serwera |
| `--api-key` | `lm-studio` | dowolny niepusty string |
| `--batch-size` | `32` | liczba tekstów na request |
| `--max-chars` | `2000` | przycięcie artykułu przed wysłaniem (większość modeli ma limit ~512 tokenów) |
| `-k`, `--clusters` | `64` | liczba klastrów |
| `--metric` | `cos` | `cos` albo `l2` |
| `--algorithm` | `auto` | nazwa algorytmu clostera (zob. `clostera.available_algorithms()`) |
| `--limit` | brak | ogranicz liczbę artykułów (debug) |
| `--skip-embed` | wył. | pomiń embedding i wczytaj `vectors.npy` z `--output-dir` (do tuningu klastrowania) |

**Przykłady:**

```bash
# pełny pipeline: embed + cluster, k=64, metric=cos
uv run 2_embed_and_cluster.py \
  -i wiki_pl.jsonl \
  --model text-embedding-bge-m3 \
  -k 64

# eksperymenty z liczbą klastrów bez ponownego embeddowania
uv run 2_embed_and_cluster.py --skip-embed -k 256 --metric cos --model unused
```

**Pliki wyjściowe** (w `--output-dir`):

- `vectors.npy` — `float32` o kształcie `(N, D)`, jeden wiersz na artykuł, w kolejności wejściowego JSONL
- `labels.npy` — etykiety klastrów (jedna na artykuł)
- `wiki_pl_clustered.jsonl` — kopia wejścia z dodanym polem `"cluster": <int>` w każdej linii
- `clusters/cluster_000.jsonl` … `clusters/cluster_NNN.jsonl` — JSONL podzielony per klaster (zero‑paddowane numery; przy każdym uruchomieniu folder jest czyszczony, żeby zmiana `-k` nie zostawiała starych plików)

**Podgląd zawartości klastra:**

```bash
# tytuły z klastra 5
jq -r '.meta.title' out/clusters/cluster_005.jsonl

# rozmiary klastrów malejąco
wc -l out/clusters/cluster_*.jsonl | sort -rn | head -20
```

### `3_label_and_visualize.py` — etykiety przez LLM + wizualizacja UMAP/plotly

Dla każdego klastra wybiera N artykułów najbliższych centroidowi (cosine), wysyła je do instruct‑modelu z prośbą o krótką polską nazwę klastra, redukuje wektory do 2D/3D przez UMAP i renderuje interaktywny wykres plotly w HTML.

**Wymagania po stronie serwera:** model instrukcyjny po polsku (np. `bielik-11b-v3.0-instruct`, `bielik-4.5b-v3.0-instruct@q8_0`).

**Argumenty CLI:**

| Flaga | Domyślnie | Opis |
|---|---|---|
| `-i`, `--input-dir` | `out` | katalog z `vectors.npy`, `labels.npy`, `*_clustered.jsonl` |
| `-o`, `--output-dir` | `=input-dir` | katalog na `cluster_labels.json` i HTML |
| `--clustered-jsonl` | (auto) | konkretna ścieżka JSONL z klastrami; domyślnie pierwszy `*_clustered.jsonl` |
| `--model` | wymagane (chyba że `--skip-labeling`) | id modelu w serwerze |
| `--base-url` | `http://localhost:1234/v1` | endpoint serwera |
| `--api-key` | `lm-studio` | token serwera |
| `--samples-per-cluster` | `8` | ile artykułów per klaster wysyła do LLM |
| `--max-chars` | `400` | przycięcie próbki przed promptem |
| `--temperature` | `0.2` | temperatura modelu |
| `--reducer` | `umap` | `umap` albo `pca` |
| `--3d` | wył. | render 3D zamiast 2D |
| `--annotate-top` | `30` | ile największych klastrów dostaje etykiety na centroidzie |
| `--plot-name` | `clusters_2d.html`/`clusters_3d.html` | nazwa pliku wyjściowego |
| `--skip-labeling` | wył. | pomiń wywołania LLM i użyj istniejącego `cluster_labels.json` |

**Przykłady:**

```bash
# pełen przebieg: nazwij klastry przez Bielika i zrób wykres 2D
uv run 3_label_and_visualize.py \
  --model bielik-11b-v3.0-instruct \
  --api-key "$LMSTUDIO_API_KEY"

# 3D, bez ponownego pytania LLM
uv run 3_label_and_visualize.py --skip-labeling --3d

# tunowanie UMAP/PCA, oddzielny plik wyjściowy
uv run 3_label_and_visualize.py --skip-labeling --reducer pca --plot-name clusters_pca.html
```

**Pliki wyjściowe:**

- `cluster_labels.json` — mapa `cluster_id → nazwa` (zapisana po wywołaniach LLM, czytana przy `--skip-labeling`)
- `clusters_2d.html` lub `clusters_3d.html` — interaktywny wykres plotly. Hover na punkt = nazwa klastra + tytuł artykułu. Klik w legendzie = ukryj/pokaż klaster, dwuklik = solo.

**Otwórz w przeglądarce** (VS Code preview słabo renderuje plotly):

```bash
open out/clusters_2d.html        # albo clusters_3d.html
```
