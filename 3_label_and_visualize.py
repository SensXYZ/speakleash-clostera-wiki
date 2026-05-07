#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "loguru",
#     "numpy",
#     "openai",
#     "plotly",
#     "python-dotenv",
#     "scikit-learn",
#     "umap-learn",
# ]
# ///
"""Label clusters via an instruct LLM (LM Studio) and visualize them with UMAP + plotly."""

import argparse
import colorsys
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv(override=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input-dir", default="out",
                   help="dir containing vectors.npy / labels.npy / *_clustered.jsonl. Default: out")
    p.add_argument("--clustered-jsonl", default=None,
                   help="path to *_clustered.jsonl. Default: first match in --input-dir")
    p.add_argument("-o", "--output-dir", default=None,
                   help="output dir for cluster_labels.json + plot. Defaults to --input-dir")
    p.add_argument("--base-url", default="http://localhost:1234/v1")
    p.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
                   help="API key for the server. Default: $LMSTUDIO_API_KEY or 'lm-studio'")
    p.add_argument("--model", default=None,
                   help="instruct model id (required unless --skip-labeling)")
    p.add_argument("--samples-per-cluster", type=int, default=8,
                   help="representative articles sent to the LLM per cluster. Default: 8")
    p.add_argument("--max-chars", type=int, default=400,
                   help="truncate each sampled article to N chars. Default: 400")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--reducer", choices=["umap", "pca"], default="umap",
                   help="2D/3D dim reduction. Default: umap")
    p.add_argument("--3d", dest="three_d", action="store_true",
                   help="render 3D scatter instead of 2D")
    p.add_argument("--annotate-top", type=int, default=30,
                   help="annotate this many largest clusters with text labels. Default: 30")
    p.add_argument("--plot-name", default=None,
                   help="output HTML file name. Default: clusters_2d.html or clusters_3d.html")
    p.add_argument("--skip-labeling", action="store_true",
                   help="reuse existing cluster_labels.json instead of querying the LLM")
    return p.parse_args()


def load_data(input_dir: Path, clustered_jsonl):
    vectors = np.load(input_dir / "vectors.npy").astype(np.float32, copy=False)
    labels = np.load(input_dir / "labels.npy")

    if clustered_jsonl is None:
        candidates = sorted(input_dir.glob("*_clustered.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"no *_clustered.jsonl in {input_dir}")
        clustered_jsonl = candidates[0]

    records = []
    with open(clustered_jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not (len(records) == vectors.shape[0] == labels.shape[0]):
        raise ValueError(
            f"length mismatch: records={len(records)} "
            f"vectors={vectors.shape[0]} labels={labels.shape[0]}"
        )
    return vectors, labels, records, clustered_jsonl


def representative_samples(vectors, labels, records, n_per_cluster):
    by_cluster = {}
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        cluster_vecs = vectors[idx]
        centroid = cluster_vecs.mean(axis=0)
        denom = (np.linalg.norm(cluster_vecs, axis=1) + 1e-9) * (np.linalg.norm(centroid) + 1e-9)
        sims = cluster_vecs @ centroid / denom
        order = np.argsort(-sims)
        picked = idx[order[:n_per_cluster]]
        by_cluster[int(k)] = [records[i] for i in picked]
    return by_cluster


def article_title(rec) -> str:
    meta = rec.get("meta") or {}
    return meta.get("title") or rec.get("title") or "(bez tytułu)"


def name_cluster(client, model, samples, max_chars, temperature) -> str:
    bullets = []
    for i, r in enumerate(samples, 1):
        title = article_title(r)
        text = (r.get("text") or "").replace("\n", " ").strip()[:max_chars]
        bullets.append(f"{i}. {title} — {text}")

    user_msg = (
        "Poniżej znajdziesz reprezentatywne artykuły z polskiej Wikipedii należące "
        "do tego samego klastra tematycznego. Nadaj klastrowi krótką, opisową nazwę "
        "po polsku (2–5 słów) ujmującą wspólny motyw.\n\n"
        + "\n".join(bullets)
        + "\n\nOdpowiedz TYLKO nazwą klastra — bez wstępu, cudzysłowów i kropki na końcu."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "Jesteś ekspertem od klasyfikacji tematycznej tekstów w języku polskim."},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=40,
    )
    name = (resp.choices[0].message.content or "").strip()
    return name.strip().strip('"').strip("'").rstrip(".")


def reduce_dims(vectors, reducer, n_components):
    if reducer == "umap":
        try:
            import umap
            logger.info("running UMAP → {}D (this takes a minute)...", n_components)
            return umap.UMAP(
                n_components=n_components,
                metric="cosine",
                n_neighbors=15,
                min_dist=0.1,
                random_state=42,
            ).fit_transform(vectors)
        except ImportError:
            logger.warning("umap-learn missing; falling back to PCA")

    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=42).fit_transform(vectors)


def discrete_palette(n):
    """n visually-distinct colors via HSV — good for k>20 where matplotlib cmaps break down."""
    out = []
    for i in range(n):
        # interleave hue so neighboring clusters get contrasting colors
        hue = ((i * 0.618033988749895) % 1.0)
        rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
        out.append(f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})")
    return out


def plot_plotly(coords, labels, names, titles, out_path, annotate_top, dim):
    import plotly.graph_objects as go

    unique, counts = np.unique(labels, return_counts=True)
    n = len(unique)
    palette = discrete_palette(n)
    largest = set(unique[np.argsort(-counts)[:annotate_top]].tolist())

    fig = go.Figure()

    # ordering: largest clusters first → they appear at top of legend
    order = np.argsort(-counts)
    titles_arr = np.asarray(titles, dtype=object)

    for slot, idx in enumerate(order):
        k = int(unique[idx])
        cluster_name = names.get(k, f"klaster {k}")
        color = palette[slot]
        mask = labels == k
        size = int(mask.sum())

        cluster_titles = titles_arr[mask]
        hover_text = [
            f"<b>{cluster_name}</b><br>klaster {k} ({size} art.)<br>{(t or '').replace('<','&lt;').replace('>','&gt;')}"
            for t in cluster_titles
        ]
        legend_name = f"{cluster_name} ({size})"

        common = dict(
            mode="markers",
            text=hover_text, hoverinfo="text",
            name=legend_name,
            legendgroup=f"c{k}",
            showlegend=True,
        )
        if dim == 3:
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                marker=dict(size=3, color=color, opacity=0.8,
                            line=dict(width=0)),
                **common,
            ))
        else:
            fig.add_trace(go.Scattergl(
                x=coords[mask, 0], y=coords[mask, 1],
                marker=dict(size=6, color=color, opacity=0.75,
                            line=dict(width=0)),
                **common,
            ))

    label_xs, label_ys, label_zs, label_texts = [], [], [], []
    for k in unique:
        if int(k) not in largest:
            continue
        mask = labels == k
        label_xs.append(float(coords[mask, 0].mean()))
        label_ys.append(float(coords[mask, 1].mean()))
        label_texts.append(names.get(int(k), str(int(k))))
        if dim == 3:
            label_zs.append(float(coords[mask, 2].mean()))

    if label_texts:
        if dim == 3:
            fig.add_trace(go.Scatter3d(
                x=label_xs, y=label_ys, z=label_zs,
                mode="text", text=[f"<b>{t}</b>" for t in label_texts],
                textfont=dict(size=12, color="black"),
                hoverinfo="skip", showlegend=False,
            ))
        else:
            for x, y, txt in zip(label_xs, label_ys, label_texts, strict=True):
                fig.add_annotation(
                    x=x, y=y, text=f"<b>{txt}</b>", showarrow=False,
                    font=dict(size=11, color="black"),
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="rgba(80,80,80,0.7)",
                    borderwidth=1, borderpad=3,
                )

    fig.update_layout(
        title=f"Polish Wikipedia clusters (k={n}, {dim}D)  —  kliknij w legendzie, żeby filtrować",
        showlegend=True,
        legend=dict(
            font=dict(size=10),
            itemsizing="constant",
            tracegroupgap=2,
            yanchor="top", y=1, xanchor="left", x=1.01,
        ),
        width=1900, height=1050,
        plot_bgcolor="white",
        margin=dict(l=10, r=320, t=50, b=10),
    )
    if dim == 2:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    else:
        fig.update_layout(scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ))

    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info("wrote {}", out_path)


def main() -> int:
    args = parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_json_path = out_dir / "cluster_labels.json"
    plot_name = args.plot_name or (
        "clusters_3d.html" if args.three_d else "clusters_2d.html"
    )
    plot_path = out_dir / plot_name

    clustered_arg = Path(args.clustered_jsonl) if args.clustered_jsonl else None
    vectors, labels, records, clustered_jsonl = load_data(in_dir, clustered_arg)
    titles = [article_title(r) for r in records]
    k = len(np.unique(labels))
    logger.info("loaded {} records, dim={}, k={} (jsonl={})", len(records), vectors.shape[1], k, clustered_jsonl)

    if args.skip_labeling:
        if not labels_json_path.exists():
            logger.error("--skip-labeling but {} not found", labels_json_path)
            return 1
        with open(labels_json_path, encoding="utf-8") as fh:
            names = {int(key): val for key, val in json.load(fh).items()}
        logger.info("loaded cluster names from {}", labels_json_path)
    else:
        if not args.model:
            logger.error("--model is required unless --skip-labeling")
            return 1
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=3)
        per_cluster = representative_samples(
            vectors, labels, records, args.samples_per_cluster
        )
        names = {}
        for cid, samples in sorted(per_cluster.items()):
            try:
                name = name_cluster(
                    client, args.model, samples, args.max_chars, args.temperature
                )
            except Exception as e:
                name = f"(błąd: {type(e).__name__})"
                logger.warning("cluster {}: {} — {}", cid, name, e)
            names[cid] = name
            logger.info("cluster {:>3}: {}", cid, name)
        with open(labels_json_path, "w", encoding="utf-8") as fh:
            json.dump({str(c): n for c, n in names.items()}, fh,
                      ensure_ascii=False, indent=2)
        logger.info("wrote {}", labels_json_path)

    n_components = 3 if args.three_d else 2
    coords = reduce_dims(vectors, args.reducer, n_components)

    plot_plotly(coords, labels, names, titles, plot_path,
                args.annotate_top, n_components)
    return 0


if __name__ == "__main__":
    sys.exit(main())
