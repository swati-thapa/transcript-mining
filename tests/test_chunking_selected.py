from pathlib import Path

from src.pipeline.run_chunking import (
    CONFIG_FILE,
    build_llm_input,
    call_llm_with_retries,
    enrich_chunks,
    load_json,
    load_prompt,
    load_yaml,
    resolve_path,
    save_json,
    validate_chunks,
)


TEST_MEETING_IDS = {
    "01KQ1267C6AA7D9B3125FEC8",
    "01KQ1DE954A807A5D2653175",
    "01KQ0F8AFF3DA34FD4580008",
}

TEST_OUTPUT_FILE = "data/interim/semantic_chunks/test_chunked_dataset.json"
TEST_CHECKPOINT_FILE = "data/interim/semantic_chunks/test_chunked_dataset_checkpoint.json"


def repair_chunk_boundaries(transcript: dict, raw_chunks: list[dict]) -> list[dict]:
    first_turn_id = transcript["turns"][0]["turn_id"]
    last_turn_id = transcript["turns"][-1]["turn_id"]

    raw_chunks = sorted(
        raw_chunks,
        key=lambda x: int(x["start_turn"]),
    )

    if int(raw_chunks[0]["start_turn"]) > first_turn_id:
        raw_chunks[0]["start_turn"] = first_turn_id

    if int(raw_chunks[-1]["end_turn"]) < last_turn_id:
        raw_chunks[-1]["end_turn"] = last_turn_id

    return raw_chunks


def main():
    config = load_yaml(CONFIG_FILE)

    input_file = resolve_path(config["paths"]["enriched_dataset"])
    prompt_file = resolve_path(config["paths"]["chunking_prompt"])
    output_file = resolve_path(TEST_OUTPUT_FILE)
    checkpoint_file = resolve_path(TEST_CHECKPOINT_FILE)

    dataset = load_json(input_file)
    prompt = load_prompt(prompt_file)

    selected_dataset = [
        transcript
        for transcript in dataset
        if transcript.get("meeting_id") in TEST_MEETING_IDS
    ]

    chunked_dataset = []
    processed_ids = set()

    if checkpoint_file.exists():
        chunked_dataset = load_json(checkpoint_file)
        processed_ids = {item["meeting_id"] for item in chunked_dataset}
        print(f"Resuming test checkpoint: {len(processed_ids)} already processed")

    for idx, transcript in enumerate(selected_dataset, start=1):
        meeting_id = transcript.get("meeting_id")

        if meeting_id in processed_ids:
            print(f"Skipping already processed: {meeting_id}")
            continue

        print(f"Test chunking {idx}/{len(selected_dataset)}: {meeting_id}")

        payload = build_llm_input(transcript)

        MAX_TURNS_FOR_TESTING = 40

        original_turn_count = len(payload["turns"])
        payload["turns"] = payload["turns"][:MAX_TURNS_FOR_TESTING]

        print(
            f"Turn cap applied: {original_turn_count} -> {len(payload['turns'])}"
        )
        llm_result = call_llm_with_retries(prompt, payload, config)

        raw_chunks = llm_result.get("semantic_chunks", [])

        if not raw_chunks:
            raise ValueError(f"{meeting_id}: model returned no semantic_chunks")

        raw_chunks = repair_chunk_boundaries(transcript, raw_chunks)

        enriched_chunks = enrich_chunks(transcript, raw_chunks)
        validate_chunks(transcript, enriched_chunks)

        transcript["semantic_chunks"] = enriched_chunks
        chunked_dataset.append(transcript)

        save_json(chunked_dataset, checkpoint_file)
        print(f"Checkpoint saved: {checkpoint_file}")

    save_json(chunked_dataset, checkpoint_file)
    save_json(chunked_dataset, output_file)

    print(f"\nSaved test chunked dataset to: {output_file}")
    print(f"Total test transcripts chunked: {len(chunked_dataset)}")

if __name__ == "__main__":
    main()