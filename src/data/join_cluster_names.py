"""
Joins cluster IDs and cluster names into the nested substantive dataset.

Builds a lookup from chunk_id to cluster_id and cluster_name using the flat
clustered chunks and the cluster name mapping. Then updates each chunk in the
nested dataset with its cluster_id and cluster_name. Chunks not found in the
mapping are assigned None for both fields.

Reads from:  data/processed/chunk_embeddings/chunked_dataset_with_embeddings_substantive.json
             data/processed/clustered_chunks/clustered_chunks_flat.json
             data/processed/clustered_chunks/cluster_name_mapping.json
Writes to:   data/processed/clustered_chunks/named_clustered_dataset.json
"""

import json
from pathlib import Path

import yaml


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


def main():
    config = load_yaml(CONFIG_FILE)

    embedded_dataset_file = resolve_path(
        config["paths"]["embedded_dataset_substantive"]
    )

    clustered_chunks_file = resolve_path(
        config["paths"]["clustered_chunks_flat"]
    )

    cluster_name_mapping_file = resolve_path(
        config["paths"]["cluster_name_mapping"]
    )

    output_file = resolve_path(
        config["paths"]["named_clustered_dataset"]
    )

    nested_dataset = load_json(embedded_dataset_file)
    clustered_chunks = load_json(clustered_chunks_file)
    cluster_name_mapping = load_json(cluster_name_mapping_file)

    cluster_id_to_name = {
        int(item["cluster_id"]): item["cluster_name"]
        for item in cluster_name_mapping
    }

    chunk_cluster_mapping = {}

    for chunk in clustered_chunks:
        chunk_id = chunk.get("chunk_id")
        cluster_id = chunk.get("cluster_id")

        if chunk_id is None or cluster_id is None:
            continue

        cluster_id = int(cluster_id)

        chunk_cluster_mapping[chunk_id] = {
            "cluster_id": cluster_id,
            "cluster_name": cluster_id_to_name.get(cluster_id),
        }

    updated_chunks = 0
    missing_chunks = 0

    for transcript in nested_dataset:
        for chunk in transcript.get("semantic_chunks", []):
            chunk_id = chunk.get("chunk_id")

            if chunk_id in chunk_cluster_mapping:
                chunk["cluster_id"] = chunk_cluster_mapping[chunk_id]["cluster_id"]
                chunk["cluster_name"] = chunk_cluster_mapping[chunk_id]["cluster_name"]
                updated_chunks += 1
            else:
                chunk["cluster_id"] = None
                chunk["cluster_name"] = None
                missing_chunks += 1

    save_json(nested_dataset, output_file)

    print(f"Updated chunks with cluster names: {updated_chunks}")
    print(f"Chunks without cluster mapping: {missing_chunks}")
    print(f"Saved named clustered dataset to: {output_file}")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()