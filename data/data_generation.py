"""
Consolidates raw meeting files into a single dataset file.

For each meeting folder in raw/dataset/ it reads five files:
meeting-info.json, summary.json, speaker-meta.json, events.json, transcript.json.
It builds a unified record with meeting metadata, summary metadata, speaker metadata,
events, and turns. Speaker roles are inferred from speaker names using keyword matching.
Action items are parsed from a "owner: text" string format.

Reads from:  raw/dataset/<meeting_folders>/
Writes to:   consolidated_dataset.json
"""

import json
from pathlib import Path


DATASET_ROOT = Path("raw/dataset")
OUTPUT_FILE = "consolidated_dataset.json"


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def infer_speaker_role(name: str) -> str:
    """
    Very lightweight role inference.
    Can improve later if needed.
    """

    customer_keywords = [
        "director",
        "manager",
        "client",
        "customer"
    ]

    lowered = name.lower()

    for keyword in customer_keywords:
        if keyword in lowered:
            return "customer"

    return "agent"


def build_turns(transcript_data, speaker_meta):
    turns = []

    speaker_id_map = {
        int(k): v for k, v in speaker_meta.items()
    }

    for idx, row in enumerate(transcript_data["data"]):

        speaker_id = row.get("speaker_id")
        speaker_name = row.get("speaker_name")

        turn = {
            "turn_id": idx,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "speaker_role": infer_speaker_role(speaker_name),
            "text": row.get("sentence"),
            "start_time": row.get("time"),
            "end_time": row.get("endTime"),
            "sentiment": row.get("sentimentType"),
            "confidence": row.get("averageConfidence")
        }

        turns.append(turn)

    return turns


def build_speakers(speaker_meta):
    speakers = []

    for speaker_id, speaker_name in speaker_meta.items():

        speaker = {
            "speaker_id": int(speaker_id),
            "speaker_name": speaker_name,
            "role": infer_speaker_role(speaker_name)
        }

        speakers.append(speaker)

    return speakers


def consolidate_meeting(meeting_folder: Path):

    meeting_info = load_json(meeting_folder / "meeting-info.json")
    summary_data = load_json(meeting_folder / "summary.json")
    speaker_meta = load_json(meeting_folder / "speaker-meta.json")
    events_data = load_json(meeting_folder / "events.json")
    transcript_data = load_json(meeting_folder / "transcript.json")

    consolidated = {

        "meeting_id": meeting_info.get("meetingId"),

        "meeting_metadata": {
            "title": meeting_info.get("title"),
            "start_time": meeting_info.get("startTime"),
            "end_time": meeting_info.get("endTime"),
            "duration_minutes": meeting_info.get("duration"),
            "organizer_email": meeting_info.get("organizerEmail"),
            "host": meeting_info.get("host"),
            "participants": meeting_info.get("allEmails", [])
        },

        "summary_metadata": {
            "summary": summary_data.get("summary"),
            "provided_topics": summary_data.get("topics", []),
            "overall_sentiment": summary_data.get("overallSentiment"),
            "sentiment_score": summary_data.get("sentimentScore"),
            "action_items": [],
            "key_moments": summary_data.get("keyMoments", [])
        },

        "speaker_metadata": {
            "speaker_id_map": speaker_meta,
            "speakers": build_speakers(speaker_meta)
        },

        "events": [],

        "turns": build_turns(
            transcript_data=transcript_data,
            speaker_meta=speaker_meta
        ),

        "semantic_chunks": []
    }

    # Action Items
    for item in summary_data.get("actionItems", []):

        if ":" in item:
            owner, text = item.split(":", 1)

            consolidated["summary_metadata"]["action_items"].append({
                "owner": owner.strip(),
                "text": text.strip()
            })

        else:
            consolidated["summary_metadata"]["action_items"].append({
                "owner": None,
                "text": item
            })

    # Events
    for event in events_data:

        consolidated["events"].append({
            "participant_name": event.get("participantName"),
            "event_type": event.get("type"),
            "time": event.get("time"),
            "timestamp": event.get("timestamp")
        })

    return consolidated


def main():

    consolidated_dataset = []

    meeting_folders = sorted([
        folder for folder in DATASET_ROOT.iterdir()
        if folder.is_dir()
    ])

    for meeting_folder in meeting_folders:

        try:
            consolidated = consolidate_meeting(meeting_folder)
            consolidated_dataset.append(consolidated)

            print(f"Processed: {meeting_folder.name}")

        except Exception as e:
            print(f"Failed processing {meeting_folder.name}: {e}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(consolidated_dataset, f, indent=2)

    print(f"\nSaved consolidated dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()