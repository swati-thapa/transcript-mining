"""
Enriches the consolidated transcript dataset with derived metadata.

For each transcript it derives:
- invitee_domains: unique email domains extracted from participant emails
- is_cross_domain: True if more than one domain is present
- call_type: customer_facing if cross-domain, internal_operational otherwise
- is_support_call: True if the meeting title contains the word "support"

Reads from:  data/consolidated_dataset.json
Writes to:   data/interim/enriched_dataset.json
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


def domain_of(email: str) -> str | None:
    if not email or "@" not in email:
        return None

    return email.split("@", 1)[1].strip().lower()


def derive_cross_domain_metadata(invitee_emails: list[str]) -> dict:
    invitee_domains = sorted(
        {
            domain_of(email)
            for email in invitee_emails
            if domain_of(email)
        }
    )

    is_cross_domain = len(invitee_domains) > 1

    if is_cross_domain:
        call_type = "customer_facing"
    else:
        call_type = "internal_operational"

    return {
        "invitee_domains": invitee_domains,
        "is_cross_domain": is_cross_domain,
        "call_type": call_type,
    }


def get_invitee_emails(transcript: dict) -> list[str]:
    """
    Supports both possible structures:

    1. Flat consolidated structure:
       transcript["invitee_emails"]

    2. Nested structure:
       transcript["meeting_metadata"]["participants"]
    """

    invitee_emails = transcript.get("invitee_emails")

    if invitee_emails:
        return invitee_emails

    meeting_metadata = transcript.get("meeting_metadata", {})

    participants = meeting_metadata.get("participants", [])

    return participants or []


def enrich_transcript(transcript: dict) -> dict:
    meeting_metadata = transcript.get("meeting_metadata", {})

    invitee_emails = get_invitee_emails(transcript)

    cross_domain_metadata = derive_cross_domain_metadata(
        invitee_emails=invitee_emails
    )

    title = meeting_metadata.get("title") or transcript.get("title", "")
    is_support_call = is_support_call_from_title(title)

    meeting_metadata["invitee_emails"] = invitee_emails
    meeting_metadata.update(cross_domain_metadata)

    meeting_metadata["is_support_call"] = is_support_call

    transcript["meeting_metadata"] = meeting_metadata

    return transcript


def enrich_dataset(dataset: list[dict]) -> list[dict]:
    enriched_dataset = []

    for transcript in dataset:
        enriched_dataset.append(enrich_transcript(transcript))

    return enriched_dataset


def is_support_call_from_title(title: str) -> bool:
    title = (title or "").lower()
    return "support" in title


def main():
    config = load_yaml(CONFIG_FILE)

    input_file = resolve_path(config["paths"]["consolidated_dataset"])
    output_file = resolve_path(config["paths"]["enriched_dataset"])

    dataset = load_json(input_file)

    enriched_dataset = enrich_dataset(dataset)

    save_json(enriched_dataset, output_file)

    print(f"Saved enriched dataset to: {output_file}")
    print(f"Total transcripts enriched: {len(enriched_dataset)}")


if __name__ == "__main__":
    main()