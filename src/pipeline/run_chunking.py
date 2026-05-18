"""
Splits each transcript into semantic chunks using an LLM.

For each transcript the turns and provided topics are sent to an LLM
(OpenAI, Groq, or Ollama, configured in config.yaml) with a chunking prompt.
The LLM returns chunk boundaries with labels, summaries, sentiment, and matched topics.
The script repairs common boundary issues (overlapping chunks, missing start/end turns,
uncovered middle gaps) and validates that all turns are covered exactly once.
Supports checkpointing to resume processing from where it stopped.

Reads from:  data/interim/enriched_dataset.json
             config/prompts/chunking_prompt.txt
Writes to:   data/interim/semantic_chunks/chunked_dataset.json
             data/interim/semantic_chunks/chunked_dataset_checkpoint.json (incremental)
"""

import json
from collections import Counter
from pathlib import Path
import os
from json_repair import repair_json
import time

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()


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


def load_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response:\n{text}")

    json_text = text[start:end + 1]

    repaired_json = repair_json(json_text)

    return json.loads(repaired_json)



def call_groq(prompt: str, payload: dict, config: dict) -> dict:
    from openai import OpenAI

    groq_config = config["chunking"]["groq"]

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=groq_config["base_url"],
    )

    response = client.chat.completions.create(
        model=groq_config["model"],
        temperature=groq_config.get("temperature", 0),
        max_tokens=groq_config.get("max_tokens", 4096),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )


    content = response.choices[0].message.content
    return extract_json(content)

def call_ollama(prompt: str, payload: dict, config: dict) -> dict:
    ollama_config = config["chunking"]["ollama"]

    response = requests.post(
        ollama_config["url"],
        json={
            "model": ollama_config["model"],
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "options": {
                "temperature": ollama_config.get("temperature", 0)
            },
        },
        timeout=ollama_config.get("timeout_seconds", 180),
    )

    response.raise_for_status()
    result = response.json()

    return extract_json(result["message"]["content"])


def call_openai(prompt: str, payload: dict, config: dict) -> dict:
    from openai import OpenAI

    openai_config = config["chunking"]["openai"]

    client = OpenAI()

    response = client.chat.completions.create(
        model=openai_config["model"],
        temperature=openai_config.get("temperature", 0),
        #response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    return json.loads(response.choices[0].message.content)


def call_llm(prompt: str, payload: dict, config: dict) -> dict:
    provider = config["chunking"]["provider"]

    if provider == "ollama":
        return call_ollama(prompt, payload, config)

    if provider == "openai":
        return call_openai(prompt, payload, config)

    if provider == "groq":
        return call_groq(prompt, payload, config)

    raise ValueError(f"Unsupported chunking provider: {provider}")


def get_turns_in_range(turns: list[dict], start_turn: int, end_turn: int) -> list[dict]:
    return [
        turn
        for turn in turns
        if start_turn <= turn.get("turn_id") <= end_turn
    ]


def build_chunk_text(turns: list[dict], start_turn: int, end_turn: int) -> str:
    selected = get_turns_in_range(turns, start_turn, end_turn)

    return "\n".join(
        f'{turn.get("speaker_name", "UNKNOWN")}: {turn.get("text", "")}'
        for turn in selected
    )


def get_chunk_times(turns: list[dict], start_turn: int, end_turn: int):
    selected = get_turns_in_range(turns, start_turn, end_turn)

    if not selected:
        return None, None

    return selected[0].get("start_time"), selected[-1].get("end_time")


def get_turn_sentiment_counts(turns: list[dict], start_turn: int, end_turn: int) -> dict:
    selected = get_turns_in_range(turns, start_turn, end_turn)

    sentiments = [
        turn.get("sentiment")
        for turn in selected
        if turn.get("sentiment")
    ]

    return dict(Counter(sentiments))


def enrich_chunks(transcript: dict, raw_chunks: list[dict]) -> list[dict]:
    meeting_id = transcript.get("meeting_id")
    turns = transcript.get("turns", [])

    enriched_chunks = []

    for idx, chunk in enumerate(raw_chunks, start=1):
        start_turn = int(chunk["start_turn"])
        end_turn = int(chunk["end_turn"])

        start_time, end_time = get_chunk_times(turns, start_turn, end_turn)

        chunk_id = chunk.get("chunk_id") or f"{meeting_id}_CHUNK_{idx:03d}"

        enriched_chunks.append(
            {
                "meeting_id": meeting_id,
                "chunk_id": chunk_id,
                "start_turn": start_turn,
                "end_turn": end_turn,
                "start_time": start_time,
                "end_time": end_time,
                "chunk_text": build_chunk_text(turns, start_turn, end_turn),
                "chunk_label": chunk.get("chunk_label"),
                "chunk_summary": chunk.get("chunk_summary"),
                "matched_provided_topics": chunk.get("matched_provided_topics", []),
                "chunk_sentiment": chunk.get("chunk_sentiment"),
                "sentiment_reason": chunk.get("sentiment_reason"),
                "turn_sentiment_counts": get_turn_sentiment_counts(
                    turns,
                    start_turn,
                    end_turn,
                ),
            }
        )

    return enriched_chunks


def validate_chunks(transcript: dict, chunks: list[dict]) -> None:
    meeting_id = transcript.get("meeting_id")
    turns = transcript.get("turns", [])

    expected_turn_ids = {turn.get("turn_id") for turn in turns}
    covered_turn_ids = []

    for chunk in chunks:
        start_turn = chunk["start_turn"]
        end_turn = chunk["end_turn"]

        if start_turn > end_turn:
            raise ValueError(
                f"{meeting_id}: invalid chunk boundary in {chunk['chunk_id']}"
            )

        covered_turn_ids.extend(range(start_turn, end_turn + 1))

    covered_set = set(covered_turn_ids)

    missing = sorted(expected_turn_ids - covered_set)
    extra = sorted(covered_set - expected_turn_ids)

    if missing:
        raise ValueError(f"{meeting_id}: missing turns in chunks: {missing}")

    if extra:
        raise ValueError(f"{meeting_id}: chunks reference invalid turns: {extra}")

    if len(covered_turn_ids) != len(set(covered_turn_ids)):
        raise ValueError(f"{meeting_id}: overlapping chunks detected")
    
def build_llm_input(transcript: dict) -> dict:
    return {
        "meeting_id": transcript.get("meeting_id"),

        "provided_topics": transcript.get(
            "summary_metadata",
            {}
        ).get("provided_topics", []),

        "turns": [
            {
                "turn_id": turn.get("turn_id"),
                "speaker_role": turn.get("speaker_role"),
                "text": turn.get("text"),
                "sentiment": turn.get("sentiment"),
            }
            for turn in transcript.get("turns", [])
        ],
    }

def call_llm_with_retries(prompt, payload, config, retries=3):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            return call_llm(prompt, payload, config)
        except Exception as e:
            last_error = e
            print(f"LLM call failed. Attempt {attempt}/{retries}. Error: {e}")

        time.sleep(5)

    raise last_error

def repair_overlapping_chunks(raw_chunks: list[dict]) -> list[dict]:
    raw_chunks = sorted(raw_chunks, key=lambda x: int(x["start_turn"]))

    repaired = []

    previous_end = None

    for chunk in raw_chunks:
        chunk = dict(chunk)

        start_turn = int(chunk["start_turn"])
        end_turn = int(chunk["end_turn"])

        if previous_end is not None and start_turn <= previous_end:
            start_turn = previous_end + 1
            chunk["start_turn"] = start_turn

        if start_turn <= end_turn:
            repaired.append(chunk)
            previous_end = end_turn

    return repaired

def repair_missing_middle_gaps(raw_chunks: list[dict]) -> list[dict]:
    raw_chunks = sorted(raw_chunks, key=lambda x: int(x["start_turn"]))

    for i in range(len(raw_chunks) - 1):
        current_end = int(raw_chunks[i]["end_turn"])
        next_start = int(raw_chunks[i + 1]["start_turn"])

        if next_start > current_end + 1:
            raw_chunks[i]["end_turn"] = next_start - 1

    return raw_chunks

def main():
    config = load_yaml(CONFIG_FILE)

    input_file = resolve_path(config["paths"]["enriched_dataset"])
    prompt_file = resolve_path(config["paths"]["chunking_prompt"])
    output_file = resolve_path(config["paths"]["chunked_dataset"])
    checkpoint_file = resolve_path(config["paths"]["chunking_checkpoint"])

    dataset = load_json(input_file)
    prompt = load_prompt(prompt_file)

    max_transcripts = config["chunking"].get("max_transcripts")
    checkpoint_every = config["chunking"].get("checkpoint_every", 1)
    resume = config["chunking"].get("resume_from_checkpoint", True)

    if max_transcripts:
        dataset = dataset[:max_transcripts]

    chunked_dataset = []
    processed_meeting_ids = set()

    if resume and checkpoint_file.exists():
        chunked_dataset = load_json(checkpoint_file)

        processed_meeting_ids = {
            item["meeting_id"]
            for item in chunked_dataset
        }

        print(
            f"Resuming from checkpoint: "
            f"{len(processed_meeting_ids)} already processed"
        )

    provider = config["chunking"]["provider"]

    print(f"Using chunking provider: {provider}")
    print(f"Processing max transcripts: {max_transcripts or 'all'}")

    processed_count = len(chunked_dataset)

    for idx, transcript in enumerate(dataset, start=1):
        meeting_id = transcript.get("meeting_id")

        if meeting_id in processed_meeting_ids:
            print(f"Skipping already processed: {meeting_id}")
            continue

        print(f"Chunking {idx}/{len(dataset)}: {meeting_id}")

        payload = build_llm_input(transcript)

        llm_result = call_llm_with_retries(
            prompt,
            payload,
            config,
        )

        raw_chunks = llm_result.get("semantic_chunks", [])

        if not raw_chunks:
            print("LLM RESULT:")
            print(json.dumps(llm_result, indent=2, ensure_ascii=False))

            raise ValueError(
                f"{meeting_id}: model returned no semantic_chunks"
            )

        first_turn_id = transcript["turns"][0]["turn_id"]
        last_turn_id = transcript["turns"][-1]["turn_id"]

        raw_chunks = sorted(
            raw_chunks,
            key=lambda x: int(x["start_turn"])
        )

        # Fix missing starting turns
        if int(raw_chunks[0]["start_turn"]) > first_turn_id:
            raw_chunks[0]["start_turn"] = first_turn_id

        # Fix missing ending turns
        if int(raw_chunks[-1]["end_turn"]) < last_turn_id:
            raw_chunks[-1]["end_turn"] = last_turn_id

        # Fix overlapping chunk boundaries
        raw_chunks = repair_overlapping_chunks(raw_chunks)

        # Fix uncovered middle gaps
        raw_chunks = repair_missing_middle_gaps(raw_chunks)

        enriched_chunks = enrich_chunks(
            transcript,
            raw_chunks,
        )

        validate_chunks(
            transcript,
            enriched_chunks,
        )
     

        transcript["semantic_chunks"] = enriched_chunks

        chunked_dataset.append(transcript)

        processed_count += 1

        if processed_count % checkpoint_every == 0:
            save_json(chunked_dataset, checkpoint_file)

            print(
                f"Checkpoint saved: {checkpoint_file}"
            )

    save_json(chunked_dataset, checkpoint_file)
    save_json(chunked_dataset, output_file)

    print(f"\nSaved final chunked dataset to: {output_file}")
if __name__ == "__main__":
    main()