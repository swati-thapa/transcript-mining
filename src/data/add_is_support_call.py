"""
Copies is_support_call from enriched_dataset.json into named_clustered_dataset.json
by matching on meeting_id, and writes the result to a new file.

Reads from:  data/interim/enriched_dataset.json (source of is_support_call)
             data/processed/clustered_chunks/named_clustered_dataset.json (input)
Writes to:   data/processed/clustered_chunks/named_clustered_dataset_with_support.json
"""

import json
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

ENRICHED_PATH = BASE_DIR / config["paths"]["enriched_dataset"]
CLUSTERED_PATH = BASE_DIR / config["paths"]["named_clustered_dataset"]
OUTPUT_PATH = BASE_DIR / config["paths"]["named_clustered_dataset_with_support"]

with open(ENRICHED_PATH) as f:
    enriched = json.load(f)

with open(CLUSTERED_PATH) as f:
    clustered = json.load(f)

# Build lookup: meeting_id -> is_support_call
support_lookup = {
    record["meeting_id"]: record["meeting_metadata"].get("is_support_call")
    for record in enriched
    if record.get("meeting_id") and record.get("meeting_metadata")
}

matched = 0
missing = []

for record in clustered:
    mid = record.get("meeting_id")
    if mid in support_lookup:
        record["meeting_metadata"]["is_support_call"] = support_lookup[mid]
        matched += 1
    else:
        missing.append(mid)

with open(OUTPUT_PATH, "w") as f:
    json.dump(clustered, f, indent=2)

print(f"Done. {matched}/{len(clustered)} records updated.")
if missing:
    print(f"No match found for {len(missing)} meeting_id(s): {missing}")
