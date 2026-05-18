"""
Generates embeddings for all semantic chunks using a SentenceTransformer model.

Flattens the nested chunked dataset into individual chunks and builds an embedding
text from each chunk's label and summary (metadata such as sentiment, call type,
and topics are excluded from the embedding text and kept only for downstream analysis).
Embeddings are generated in batches using BAAI/bge-large-en-v1.5 (configurable).
The embeddings are attached back to both the flat chunk list and the nested dataset.

Reads from:  data/interim/semantic_chunks/chunked_dataset.json
Writes to:   data/processed/chunk_embeddings/chunked_dataset_with_embeddings.json
             data/processed/chunk_embeddings/chunk_embeddings_flat.json
"""

import json
import re
import unicodedata
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer


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


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_embedding_text(chunk: dict) -> str:
    """
    Build semantic representation for clustering.

    We use:
    - chunk_label
    - chunk_summary

    We do NOT include:
    - call_type
    - is_cross_domain
    - sentiment
    - meeting_id
    - matched topics

    Those fields are metadata for downstream analysis, not clustering.
    """

    chunk_label = normalize_text(chunk.get("chunk_label", ""))
    chunk_summary = normalize_text(chunk.get("chunk_summary", ""))

    parts = []

    if chunk_label:
        parts.append(chunk_label)

    if chunk_summary:
        parts.append(chunk_summary)

    embedding_text = ". ".join(parts).strip()

    if not embedding_text:
        raise ValueError(
            f"Missing embedding text for chunk_id={chunk.get('chunk_id')}"
        )

    return embedding_text


def flatten_chunks(dataset: list[dict]) -> list[dict]:
    flat_chunks = []

    for transcript in dataset:
        meeting_id = transcript.get("meeting_id")
        meeting_metadata = transcript.get("meeting_metadata", {})
        meeting_title = meeting_metadata.get("title")

        for chunk in transcript.get("semantic_chunks", []):
            embedding_text = build_embedding_text(chunk)

            flat_chunks.append(
                {
                    "meeting_id": meeting_id,
                    "meeting_title": meeting_title,
                    "chunk_id": chunk.get("chunk_id"),
                    "start_turn": chunk.get("start_turn"),
                    "end_turn": chunk.get("end_turn"),
                    "start_time": chunk.get("start_time"),
                    "end_time": chunk.get("end_time"),
                    "chunk_label": chunk.get("chunk_label"),
                    "chunk_summary": chunk.get("chunk_summary"),
                    "chunk_text": chunk.get("chunk_text"),
                    "embedding_text": embedding_text,

                    # metadata kept for downstream analysis only
                    "chunk_sentiment": chunk.get("chunk_sentiment"),
                    "sentiment_reason": chunk.get("sentiment_reason"),
                    "turn_sentiment_counts": chunk.get("turn_sentiment_counts", {}),
                    "matched_provided_topics": chunk.get("matched_provided_topics", []),
                    "is_cross_domain": meeting_metadata.get("is_cross_domain"),
                    "call_type": meeting_metadata.get("call_type"),
                    "invitee_domains": meeting_metadata.get("invitee_domains", []),
                }
            )

    return flat_chunks


def attach_embeddings_to_nested_dataset(
    dataset: list[dict],
    embedding_lookup: dict,
) -> list[dict]:
    for transcript in dataset:
        meeting_metadata = transcript.get("meeting_metadata", {})

        for chunk in transcript.get("semantic_chunks", []):
            chunk_id = chunk.get("chunk_id")

            # propagate metadata to chunk for easier downstream slicing
            chunk["is_cross_domain"] = meeting_metadata.get("is_cross_domain")
            chunk["call_type"] = meeting_metadata.get("call_type")
            chunk["invitee_domains"] = meeting_metadata.get("invitee_domains", [])

            if chunk_id in embedding_lookup:
                chunk["embedding_model"] = embedding_lookup[chunk_id]["embedding_model"]
                chunk["embedding_text"] = embedding_lookup[chunk_id]["embedding_text"]
                chunk["embedding"] = embedding_lookup[chunk_id]["embedding"]

    return dataset


def main():
    config = load_yaml(CONFIG_FILE)

    input_file = resolve_path(config["paths"]["chunked_dataset"])
    output_file = resolve_path(config["paths"]["embedded_dataset"])
    flat_output_file = resolve_path(config["paths"]["flat_embeddings"])

    embedding_config = config["embedding"]
    model_name = embedding_config.get("model", "BAAI/bge-large-en-v1.5")
    normalize_embeddings = embedding_config.get("normalize_embeddings", True)
    batch_size = embedding_config.get("batch_size", 32)

    dataset = load_json(input_file)
    flat_chunks = flatten_chunks(dataset)

    if not flat_chunks:
        raise ValueError("No semantic chunks found for embedding.")

    print(f"Total chunks to embed: {len(flat_chunks)}")
    print(f"Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    texts = [chunk["embedding_text"] for chunk in flat_chunks]

    print(f"Generating embeddings for {len(texts)} chunks")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )

    embedding_lookup = {}

    for chunk, embedding in zip(flat_chunks, embeddings):
        embedding_list = embedding.tolist()

        chunk["embedding_model"] = model_name
        chunk["embedding"] = embedding_list

        embedding_lookup[chunk["chunk_id"]] = {
            "embedding_model": model_name,
            "embedding_text": chunk["embedding_text"],
            "embedding": embedding_list,
        }

    embedded_dataset = attach_embeddings_to_nested_dataset(
        dataset=dataset,
        embedding_lookup=embedding_lookup,
    )

    save_json(embedded_dataset, output_file)
    save_json(flat_chunks, flat_output_file)

    print(f"Saved nested embedded dataset to: {output_file}")
    print(f"Saved flat chunk embeddings to: {flat_output_file}")
    print(f"Total chunks embedded: {len(flat_chunks)}")
    print(f"Embedding model used: {model_name}")


if __name__ == "__main__":
    main()