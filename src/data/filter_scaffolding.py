"""
Classifies chunks as scaffolding or substantive and splits the dataset accordingly.

A chunk is classified as scaffolding if:
- Its label matches a hard regex pattern (e.g. meeting setup, wrap-up, agenda, closure)
- It is short (<300 chars) and appears at the very start or end of the transcript
- It matches a soft label pattern (e.g. introduction, recap) and is short and near a boundary

Both the flat embeddings file and the nested dataset are split:
- Substantive chunks are kept for clustering
- Scaffolding chunks are saved separately

Reads from:  data/processed/chunk_embeddings/chunk_embeddings_flat.json
             data/processed/chunk_embeddings/chunked_dataset_with_embeddings.json
Writes to:   data/processed/chunk_embeddings/chunk_embeddings_flat_substantive.json
             data/processed/chunk_embeddings/chunk_embeddings_flat_scaffolding.json
             data/processed/chunk_embeddings/chunked_dataset_with_embeddings_substantive.json
"""

import json
import re
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"


HARD_SCAFFOLDING_PATTERNS = [
    r"^meeting\s+(introduction|setup|closure|wrap[- ]?up|administration|opening|close|end)$",
    r"^session\s+(introduction|opening|closure|wrap[- ]?up)$",
    r"^(introductions?|kickoff|kick[- ]?off|opening|opening\s+remarks)$",
    r"^(closure|closing|closing\s+remarks|sign[- ]?off|wrap[- ]?up)$",
    r"^action\s+items?\s+recap$",
    r"^(introductions?\s+and\s+context|introductions?\s+and\s+agenda)$",
    r"^(wrap[- ]?up\s+and\s+acknowledg(e)?ments?)$",
    r"^team\s+acknowledg(e)?ment$",
    r"^(case|call)\s+(closure|closing|wrap[- ]?up)$",
    r"^next\s+steps\s+confirmation$",
    r"^follow[- ]?up\s+actions$",
    r"^(q\d\s+)?review\s+agenda$",
    r"^agenda(\s+setting)?$",
]

SOFT_SCAFFOLDING_PATTERNS = [
    r"\bintroduction\b",
    r"\bagenda\b",
    r"\bwrap[- ]?up\b",
    r"\bclosure\b",
    r"\backnowledg(e)?ment\b",
    r"\brecap\b",
]


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


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").lower().strip())


def is_scaffolding(chunk: dict, total_turns: int) -> tuple[bool, str]:
    label = normalize_label(chunk.get("chunk_label", ""))
    text_len = len(chunk.get("chunk_text", "") or "")
    start_turn = int(chunk.get("start_turn", 0))
    rel_pos = start_turn / max(total_turns, 1)

    for pattern in HARD_SCAFFOLDING_PATTERNS:
        if re.match(pattern, label):
            return True, f"hard_label_match:{pattern}"

    if text_len < 300 and (rel_pos > 0.90 or rel_pos < 0.05):
        return True, "short_boundary_chunk"

    soft_match = any(
        re.search(pattern, label)
        for pattern in SOFT_SCAFFOLDING_PATTERNS
    )

    if soft_match and text_len < 800 and (rel_pos > 0.85 or rel_pos < 0.10):
        return True, "soft_label_boundary_short"

    return False, "substantive"


def get_total_turns_by_meeting(nested_dataset: list[dict]) -> dict:
    total_turns_by_meeting = {}

    for transcript in nested_dataset:
        meeting_id = transcript.get("meeting_id")
        turns = transcript.get("turns", [])

        if meeting_id:
            total_turns_by_meeting[meeting_id] = len(turns)

    return total_turns_by_meeting


def main():
    config = load_yaml(CONFIG_FILE)

    flat_embeddings_file = resolve_path(config["paths"]["flat_embeddings"])
    embedded_dataset_file = resolve_path(config["paths"]["embedded_dataset"])

    substantive_output_file = resolve_path(
        config["paths"]["flat_embeddings_substantive"]
    )
    scaffolding_output_file = resolve_path(
        config["paths"]["flat_embeddings_scaffolding"]
    )
    embedded_dataset_substantive_file = resolve_path(
        config["paths"]["embedded_dataset_substantive"]
    )

    flat_chunks = load_json(flat_embeddings_file)
    nested_dataset = load_json(embedded_dataset_file)

    total_turns_by_meeting = get_total_turns_by_meeting(nested_dataset)

    substantive_chunks = []
    scaffolding_chunks = []
    substantive_chunk_ids = set()

    for chunk in flat_chunks:
        meeting_id = chunk.get("meeting_id")
        total_turns = total_turns_by_meeting.get(meeting_id, 1)

        flag, reason = is_scaffolding(chunk, total_turns)

        updated_chunk = dict(chunk)
        updated_chunk["is_scaffolding"] = flag
        updated_chunk["scaffolding_reason"] = reason

        if flag:
            scaffolding_chunks.append(updated_chunk)
        else:
            substantive_chunks.append(updated_chunk)
            substantive_chunk_ids.add(chunk.get("chunk_id"))

    nested_substantive_dataset = []

    for transcript in nested_dataset:
        updated_transcript = dict(transcript)

        updated_semantic_chunks = []

        for chunk in transcript.get("semantic_chunks", []):
            chunk_id = chunk.get("chunk_id")

            if chunk_id in substantive_chunk_ids:
                updated_chunk = dict(chunk)
                updated_chunk["is_scaffolding"] = False
                updated_chunk["scaffolding_reason"] = "substantive"
                updated_semantic_chunks.append(updated_chunk)

        updated_transcript["semantic_chunks"] = updated_semantic_chunks
        nested_substantive_dataset.append(updated_transcript)

    save_json(substantive_chunks, substantive_output_file)
    save_json(scaffolding_chunks, scaffolding_output_file)
    save_json(nested_substantive_dataset, embedded_dataset_substantive_file)

    print(f"Input chunks: {len(flat_chunks)}")
    print(f"Substantive chunks: {len(substantive_chunks)}")
    print(f"Scaffolding chunks: {len(scaffolding_chunks)}")
    print(f"Saved substantive chunks to: {substantive_output_file}")
    print(f"Saved scaffolding chunks to: {scaffolding_output_file}")
    print(f"Saved nested substantive dataset to: {embedded_dataset_substantive_file}")


if __name__ == "__main__":
    main()