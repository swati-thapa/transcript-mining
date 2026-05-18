"""
Generates a human-readable name for each semantic cluster using OpenAI.

For each cluster it builds a prompt containing the top chunk labels, sentiment
distribution, call type distribution, and up to 8 representative chunk summaries
(the chunks closest to the cluster centroid). The prompt is sent to OpenAI, which
returns a short concise cluster name.

Reads from:  data/processed/clustered_chunks/clustered_chunks_flat.json
             config/prompts/cluster_naming_prompt.txt
Writes to:   data/processed/clustered_chunks/cluster_name_mapping.json
"""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"

load_dotenv()


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


def load_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_counter(counter: Counter, top_k: int = 10) -> str:
    if not counter:
        return "- none"

    return "\n".join(
        f"- {label}: {count}"
        for label, count in counter.most_common(top_k)
    )


def get_representative_summaries(chunks: list[dict], top_k: int = 8) -> list[str]:
    embeddings = np.array([chunk["embedding"] for chunk in chunks])

    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    similarities = embeddings @ centroid

    top_indices = similarities.argsort()[::-1][:top_k]

    summaries = []

    for idx in top_indices:
        summary = chunks[idx].get("chunk_summary")
        label = chunks[idx].get("chunk_label")

        if summary:
            summaries.append(f"{label}: {summary}")

    return summaries


def build_cluster_prompt(
    prompt_template: str,
    cluster_id: int,
    chunks: list[dict],
) -> str:
    top_labels = Counter(
        chunk.get("chunk_label")
        for chunk in chunks
        if chunk.get("chunk_label")
    )

    sentiment_dist = Counter(
        chunk.get("chunk_sentiment")
        for chunk in chunks
        if chunk.get("chunk_sentiment")
    )

    call_type_dist = Counter(
        chunk.get("call_type")
        for chunk in chunks
        if chunk.get("call_type")
    )

    sample_summaries = get_representative_summaries(chunks)

    return prompt_template.format(
        cluster_id=cluster_id,
        chunk_count=len(chunks),
        top_labels=format_counter(top_labels),
        sample_summaries="\n".join(f"- {s}" for s in sample_summaries),
        sentiment_dist=format_counter(sentiment_dist),
        call_type_dist=format_counter(call_type_dist),
    )


def clean_cluster_name(name: str) -> str:
    name = name.strip()
    name = name.strip('"').strip("'")
    name = name.rstrip(".")
    return name


def call_openai(prompt: str, config: dict) -> str:
    naming_config = config["naming"]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=naming_config.get("model", "gpt-4o-mini"),
        temperature=naming_config.get("temperature", 0),
        max_tokens=naming_config.get("max_tokens", 20),
        messages=[
            {
                "role": "system",
                "content": "You generate concise business taxonomy names.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return clean_cluster_name(response.choices[0].message.content)


def main():
    config = load_yaml(CONFIG_FILE)

    clustered_chunks_file = resolve_path(config["paths"]["clustered_chunks_flat"])
    naming_prompt_file = resolve_path(config["paths"]["cluster_naming_prompt"])
    output_file = resolve_path(config["paths"]["cluster_name_mapping"])

    chunks = load_json(clustered_chunks_file)
    prompt_template = load_prompt(naming_prompt_file)

    clusters = {}

    for chunk in chunks:
        cluster_id = chunk.get("cluster_id")

        if cluster_id is None:
            continue

        clusters.setdefault(int(cluster_id), []).append(chunk)

    cluster_names = []

    for cluster_id in sorted(clusters):
        cluster_chunks = clusters[cluster_id]

        print(f"Naming cluster {cluster_id} with {len(cluster_chunks)} chunks")

        prompt = build_cluster_prompt(
            prompt_template=prompt_template,
            cluster_id=cluster_id,
            chunks=cluster_chunks,
        )

        cluster_name = call_openai(prompt, config)

        cluster_names.append(
            {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
            }
        )

        print(f"Cluster {cluster_id}: {cluster_name}")

    save_json(cluster_names, output_file)

    print(f"\nSaved cluster name mapping to: {output_file}")


if __name__ == "__main__":
    main()