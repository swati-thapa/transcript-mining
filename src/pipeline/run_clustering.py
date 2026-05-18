"""
Clusters substantive chunks using agglomerative clustering on their embeddings.

Builds an embedding matrix from the flat substantive chunks and runs agglomerative
clustering with cosine distance and average linkage (n_clusters, metric, and linkage
are all configurable in config.yaml). Attaches cluster IDs to both the flat chunks
and the nested dataset. Also generates a cluster summary with top labels, sentiment
distribution, top matched topics, and sample chunks per cluster.

Reads from:  data/processed/chunk_embeddings/chunk_embeddings_flat_substantive.json
             data/processed/chunk_embeddings/chunked_dataset_with_embeddings.json
Writes to:   data/processed/clustered_chunks/clustered_chunks_flat.json
             data/processed/clustered_chunks/chunked_dataset_with_clusters.json
             data/processed/clustered_chunks/cluster_summary.json
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import AgglomerativeClustering


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    return BASE_DIR / path_str


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_embedding_matrix(flat_chunks: list[dict]) -> np.ndarray:
    embeddings = []

    for chunk in flat_chunks:
        embedding = chunk.get("embedding")

        if not embedding:
            raise ValueError(f"Missing embedding for chunk_id={chunk.get('chunk_id')}")

        embeddings.append(embedding)

    return np.array(embeddings)


def run_agglomerative_clustering(
    embedding_matrix: np.ndarray,
    clustering_config: dict,
) -> np.ndarray:
    n_clusters = clustering_config.get("n_clusters", 10)
    metric = clustering_config.get("metric", "cosine")
    linkage = clustering_config.get("linkage", "average")

    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage,
    )

    return clusterer.fit_predict(embedding_matrix)


def attach_clusters_to_flat_chunks(
    flat_chunks: list[dict],
    labels: np.ndarray,
    clustering_config: dict,
) -> list[dict]:
    method = clustering_config.get("method", "agglomerative")

    clustered_chunks = []

    for chunk, label in zip(flat_chunks, labels):
        updated_chunk = dict(chunk)

        updated_chunk["cluster_id"] = int(label)
        updated_chunk["cluster_name"] = None
        updated_chunk["clustering_method"] = method

        clustered_chunks.append(updated_chunk)

    return clustered_chunks


def attach_clusters_to_nested_dataset(
    nested_dataset: list[dict],
    clustered_chunks: list[dict],
) -> list[dict]:
    cluster_lookup = {
        chunk["chunk_id"]: {
            "cluster_id": chunk["cluster_id"],
            "cluster_name": chunk["cluster_name"],
            "clustering_method": chunk["clustering_method"],
        }
        for chunk in clustered_chunks
    }

    for transcript in nested_dataset:
        for chunk in transcript.get("semantic_chunks", []):
            chunk_id = chunk.get("chunk_id")

            if chunk_id in cluster_lookup:
                chunk["cluster_id"] = cluster_lookup[chunk_id]["cluster_id"]
                chunk["cluster_name"] = cluster_lookup[chunk_id]["cluster_name"]
                chunk["clustering_method"] = cluster_lookup[chunk_id][
                    "clustering_method"
                ]

    return nested_dataset


def summarize_clusters(clustered_chunks: list[dict]) -> list[dict]:
    cluster_ids = sorted({chunk["cluster_id"] for chunk in clustered_chunks})

    summaries = []

    for cluster_id in cluster_ids:
        members = [
            chunk for chunk in clustered_chunks
            if chunk["cluster_id"] == cluster_id
        ]

        label_counts = Counter(
            chunk.get("chunk_label")
            for chunk in members
            if chunk.get("chunk_label")
        )

        sentiment_counts = Counter(
            chunk.get("chunk_sentiment")
            for chunk in members
            if chunk.get("chunk_sentiment")
        )

        topic_counts = Counter()

        for chunk in members:
            topics = chunk.get("matched_provided_topics", [])
            if isinstance(topics, list):
                topic_counts.update(topics)

        sample_chunks = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "meeting_id": chunk.get("meeting_id"),
                "chunk_label": chunk.get("chunk_label"),
                "chunk_summary": chunk.get("chunk_summary"),
                "chunk_sentiment": chunk.get("chunk_sentiment"),
            }
            for chunk in members[:5]
        ]

        summaries.append(
            {
                "cluster_id": cluster_id,
                "chunk_count": len(members),
                "top_chunk_labels": dict(label_counts.most_common(10)),
                "sentiment_distribution": dict(sentiment_counts),
                "top_matched_provided_topics": dict(topic_counts.most_common(10)),
                "sample_chunks": sample_chunks,
            }
        )

    return summaries




def main():
    config = load_yaml(CONFIG_FILE)

    flat_embeddings_file = resolve_path(config["paths"]["flat_embeddings_substantive"])
    embedded_dataset_file = resolve_path(config["paths"]["embedded_dataset"])

    clustered_flat_file = resolve_path(
        config["paths"]["clustered_chunks_flat"]
    )
    clustered_nested_file = resolve_path(
        config["paths"]["chunked_dataset_with_clusters"]
    )
    cluster_summary_file = resolve_path(
        config["paths"]["cluster_summary"]
    )

    clustering_config = config["clustering"]

    flat_chunks = load_json(flat_embeddings_file)
    nested_dataset = load_json(embedded_dataset_file)

    embedding_matrix = build_embedding_matrix(flat_chunks)

    print(f"Loaded chunks: {len(flat_chunks)}")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Running clustering with config: {clustering_config}")

    labels = run_agglomerative_clustering(
        embedding_matrix=embedding_matrix,
        clustering_config=clustering_config,
    )

    clustered_chunks = attach_clusters_to_flat_chunks(
        flat_chunks=flat_chunks,
        labels=labels,
        clustering_config=clustering_config,
    )

    clustered_nested_dataset = attach_clusters_to_nested_dataset(
        nested_dataset=nested_dataset,
        clustered_chunks=clustered_chunks,
    )

    cluster_summary = summarize_clusters(clustered_chunks)

    save_json(clustered_chunks, clustered_flat_file)
    save_json(clustered_nested_dataset, clustered_nested_file)
    save_json(cluster_summary, cluster_summary_file)

    print(f"Saved clustered flat chunks to: {clustered_flat_file}")
    print(f"Saved nested clustered dataset to: {clustered_nested_file}")
    print(f"Saved cluster summary to: {cluster_summary_file}")


if __name__ == "__main__":
    main()