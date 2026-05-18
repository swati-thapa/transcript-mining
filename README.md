# Transcript Mining Pipeline

A pipeline for ingesting raw meeting transcripts, segmenting them into semantic chunks, embedding and clustering those chunks, and analysing sentiment and themes across call types.

---

## Pipeline Overview

```
Raw meeting files
       │
       ▼
data/data_generation.py            ← consolidate raw files into one dataset
       │
       ▼
src/data/enrich_metadata.py        ← derive call_type, is_support_call, domains
       │
       ▼
src/pipeline/run_chunking.py       ← LLM splits each transcript into semantic chunks
       │
       ▼
src/pipeline/run_embeddings.py     ← embed each chunk using SentenceTransformer
       │
       ▼
src/data/filter_scaffolding.py     ← separate substantive chunks from scaffolding
       │
       ▼
src/pipeline/run_clustering.py     ← agglomerative clustering on chunk embeddings
       │
       ▼
src/clustering/cluster_labeler.py  ← OpenAI names each cluster
       │
       ▼
src/data/join_cluster_names.py     ← attach cluster names back to nested dataset
       │
       ▼
src/data/add_is_support_call.py    ← propagate is_support_call into final dataset
       │
       ▼
named_clustered_dataset_with_support.json  ← final analysis-ready dataset
```

---

## Directory Structure

```
transcript-mining/
├── config/
│   ├── config.yaml                      # All file paths, model settings, clustering config
│   └── prompts/
│       ├── chunking_prompt.txt          # System prompt for LLM semantic chunking
│       └── cluster_naming_prompt.txt    # System prompt for cluster naming
│
├── data/
│   ├── data_generation.py               # Step 1 — consolidates raw files
│   ├── raw/dataset/<meeting_id>/        # Input: one folder per meeting
│   │   ├── meeting-info.json
│   │   ├── summary.json
│   │   ├── speaker-meta.json
│   │   ├── events.json
│   │   └── transcript.json
│   ├── consolidated_dataset.json        # Output of Step 1
│   ├── interim/
│   │   ├── enriched_dataset.json                    # Output of Step 2
│   │   └── semantic_chunks/
│   │       ├── chunked_dataset.json                 # Output of Step 3
│   │       └── chunked_dataset_checkpoint.json      # Checkpoint for resuming Step 3
│   └── processed/
│       ├── chunk_embeddings/
│       │   ├── chunked_dataset_with_embeddings.json
│       │   ├── chunk_embeddings_flat.json
│       │   ├── chunked_dataset_with_embeddings_substantive.json
│       │   ├── chunk_embeddings_flat_substantive.json
│       │   └── chunk_embeddings_flat_scaffolding.json
│       └── clustered_chunks/
│           ├── clustered_chunks_flat.json
│           ├── chunked_dataset_with_clusters.json
│           ├── cluster_summary.json
│           ├── cluster_name_mapping.json
│           ├── named_clustered_dataset.json
│           └── named_clustered_dataset_with_support.json  ← final output
│
├── src/
│   ├── data/
│   │   ├── enrich_metadata.py       # Step 2 — derives call_type, is_support_call
│   │   ├── filter_scaffolding.py    # Step 5 — splits substantive vs scaffolding chunks
│   │   ├── join_cluster_names.py    # Step 8 — attaches cluster names to nested dataset
│   │   └── add_is_support_call.py   # Step 9 — propagates is_support_call to final dataset
│   ├── pipeline/
│   │   ├── run_chunking.py          # Step 3 — LLM-based semantic chunking
│   │   ├── run_embeddings.py        # Step 4 — generate SentenceTransformer embeddings
│   │   └── run_clustering.py        # Step 6 — agglomerative clustering
│   └── clustering/
│       └── cluster_labeler.py       # Step 7 — OpenAI names each cluster
│
├── notebooks/
│   ├── 01_chunking_review.ipynb            # Inspect and validate chunking output
│   ├── 02_embedding_review.ipynb           # Explore embedding quality and distributions
│   ├── 03_clustering_experiments.ipynb     # Tune and visualise clusters
│   ├── 04_sentiment_theme_analysis.ipynb   # Sentiment analysis across call types
│   └── 05_topic_analysis.ipynb             # Topic coverage and matched topic analysis
│
└── tests/
    ├── test_chunking_selected.py    # Unit tests for chunking logic
    └── test_consolidate.py          # Unit tests for data consolidation
```

---

## Step-by-Step Description

### Step 1 — Consolidate (`data/data_generation.py`)
Reads the five raw JSON files from each meeting folder and merges them into a single unified record. Infers speaker roles (agent/customer) from speaker names. Outputs `consolidated_dataset.json`.

### Step 2 — Enrich Metadata (`src/data/enrich_metadata.py`)
Derives three fields per meeting from participant emails and the meeting title:
- `invitee_domains` — unique email domains of all participants
- `is_cross_domain` / `call_type` — `customer_facing` if multiple domains present, `internal_operational` otherwise
- `is_support_call` — `True` if the word "support" appears in the meeting title

### Step 3 — Semantic Chunking (`src/pipeline/run_chunking.py`)
Sends each transcript (turns + provided topics) to an LLM (OpenAI, Groq, or Ollama — configured in `config.yaml`). The LLM segments the transcript into semantic chunks, each with a label, summary, sentiment, and matched topics. The script repairs common boundary issues (overlaps, missing start/end, mid-meeting gaps) and validates full turn coverage. Supports checkpointing to resume from where it left off.

### Step 4 — Embeddings (`src/pipeline/run_embeddings.py`)
Embeds each chunk using `BAAI/bge-large-en-v1.5` (via SentenceTransformer). The embedding text is built from `chunk_label` + `chunk_summary` only — metadata fields like sentiment and call type are intentionally excluded so clustering is driven by semantic content, not call metadata. Outputs both a flat list and a nested dataset.

### Step 5 — Filter Scaffolding (`src/data/filter_scaffolding.py`)
Classifies chunks as scaffolding or substantive using regex patterns and positional/length heuristics. Scaffolding (introductions, wrap-ups, agendas, closings) is separated so it does not pollute clustering. Only substantive chunks are carried forward to Steps 6 onwards.

### Step 6 — Clustering (`src/pipeline/run_clustering.py`)
Runs agglomerative clustering (cosine distance, average linkage, 20 clusters by default) on the substantive chunk embeddings. Attaches cluster IDs to both the flat and nested datasets. Also generates a `cluster_summary.json` with top labels, sentiment distribution, and sample chunks per cluster.

### Step 7 — Cluster Naming (`src/clustering/cluster_labeler.py`)
For each cluster, selects the 8 chunks closest to the centroid and sends their labels, summaries, and distribution stats to OpenAI. OpenAI returns a short business-oriented name (e.g. "Billing Dispute", "SLA Risk"). Outputs `cluster_name_mapping.json`.

### Step 8 — Join Cluster Names (`src/data/join_cluster_names.py`)
Joins cluster IDs and human-readable names back into the nested substantive dataset by matching on `chunk_id`. Produces `named_clustered_dataset.json`.

### Step 9 — Add Support Flag (`src/data/add_is_support_call.py`)
Copies `is_support_call` from `enriched_dataset.json` into `named_clustered_dataset.json` by matching on `meeting_id`. Produces the final analysis-ready file: `named_clustered_dataset_with_support.json`.

---

## Configuration

All file paths and model settings live in `config/config.yaml`. No hardcoded paths exist in any script.

Key settings:

| Section | Key | Default |
|---|---|---|
| `chunking.provider` | LLM provider for chunking | `openai` |
| `chunking.max_transcripts` | Cap on transcripts to process | `100` |
| `embedding.model` | SentenceTransformer model | `BAAI/bge-large-en-v1.5` |
| `clustering.n_clusters` | Number of clusters | `20` |
| `clustering.metric` | Distance metric | `cosine` |
| `clustering.linkage` | Linkage method | `average` |
| `naming.model` | OpenAI model for cluster naming | `gpt-4o-mini` |

---

## Environment Variables

Create a `.env` file at the project root:

```
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here        # only required if using groq provider
```

---

## Running the Pipeline

Run each step in order from the project root:

```bash
python data/data_generation.py
python src/data/enrich_metadata.py
python src/pipeline/run_chunking.py
python src/pipeline/run_embeddings.py
python src/data/filter_scaffolding.py
python src/pipeline/run_clustering.py
python src/clustering/cluster_labeler.py
python src/data/join_cluster_names.py
python src/data/add_is_support_call.py
```
