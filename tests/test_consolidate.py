import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CONSOLIDATED_FILE = BASE_DIR / "data" / "consolidated_dataset.json"



REQUIRED_TOP_LEVEL_FIELDS = [
    "meeting_id",
    "meeting_metadata",
    "summary_metadata",
    "speaker_metadata",
    "events",
    "turns",
    "semantic_chunks"
]


def validate_turn(turn, transcript_index):

    required_turn_fields = [
        "turn_id",
        "speaker_id",
        "speaker_name",
        "speaker_role",
        "text",
        "start_time",
        "end_time",
        "sentiment",
        "confidence"
    ]

    for field in required_turn_fields:

        if field not in turn:
            print(
                f"[ERROR] Transcript {transcript_index} "
                f"missing turn field: {field}"
            )

    if not turn.get("text"):
        print(
            f"[ERROR] Transcript {transcript_index} "
            f"contains empty turn text"
        )


def validate_transcript(transcript, transcript_index):

    print(f"\nValidating transcript #{transcript_index}")

    # ----------------------------
    # Top-level fields
    # ----------------------------
    for field in REQUIRED_TOP_LEVEL_FIELDS:

        if field not in transcript:
            print(f"[ERROR] Missing top-level field: {field}")

    # ----------------------------
    # meeting_id
    # ----------------------------
    if not transcript.get("meeting_id"):
        print("[ERROR] meeting_id is missing or empty")

    # ----------------------------
    # turns
    # ----------------------------
    turns = transcript.get("turns", [])

    if not turns:
        print("[ERROR] No turns found")

    else:

        previous_turn_id = -1
        previous_start_time = -1

        for turn in turns:

            validate_turn(turn, transcript_index)

            current_turn_id = turn.get("turn_id", -1)
            current_start_time = turn.get("start_time", -1)

            # turn_id ordering check
            if current_turn_id <= previous_turn_id:
                print(
                    f"[ERROR] turn_id ordering issue "
                    f"at turn_id={current_turn_id}"
                )

            # time ordering check
            if current_start_time < previous_start_time:
                print(
                    f"[ERROR] start_time ordering issue "
                    f"at turn_id={current_turn_id}"
                )

            previous_turn_id = current_turn_id
            previous_start_time = current_start_time

    # ----------------------------
    # summary validation
    # ----------------------------
    summary_metadata = transcript.get("summary_metadata", {})

    if not summary_metadata.get("summary"):
        print("[WARNING] Missing summary")

    if not summary_metadata.get("provided_topics"):
        print("[WARNING] Missing provided_topics")

    # ----------------------------
    # speaker validation
    # ----------------------------
    speaker_metadata = transcript.get("speaker_metadata", {})

    if not speaker_metadata.get("speakers"):
        print("[WARNING] No speakers found")

    print("[DONE]")


def main():

    file_path = Path(CONSOLIDATED_FILE)

    if not file_path.exists():
        print(f"[ERROR] File not found: {CONSOLIDATED_FILE}")
        return

    with open(file_path, "r") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} transcripts")

    for idx, transcript in enumerate(dataset):
        validate_transcript(transcript, idx)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()