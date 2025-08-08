# TopGPT

End-to-end, hands-on pipeline to build a style-aligned language model in the voice of Andrew Tate. The project progresses in clear phases: a proof-of-concept pretraining pass (scalable to GPT‑2 774M), continued pretraining on long‑form transcripts, supervised finetuning (SFT) via synthetic Q&A generation and verification, and a planned DPO stage on a custom preference dataset.

Status
- Proof of Concept (pretraining): Done
- Continued pretraining: Done (data tools + notebook)
- Supervised finetuning: Done (data pipeline + notebooks)
- Direct Preference Optimization (DPO): Planned (dataset scaffolding present)

Why this repo
- Practice full stack: data collection, curation, training, and evaluation
- Explore scaling from a minimal GPT to larger (GPT‑2 774M‑like) configs
- Build an Andrew Tate–style model through progressively stronger supervision

Repository layout (high level)
- pretraining/
	- data/shakespeare/: tiny dataset + helper script
	- src/: minimal GPT, config, sampling, and an example training notebook
- continued_pretraining/
	- data/: YouTube link fetcher, transcript extractor, helper notebooks
	- src/: continued pretraining notebook
- supervised_finetuning/
	- data/: generated/verified datasets
	- data_processing/: PDF→Q&A generation, preprocessing, verification
	- src/: SFT and inference notebooks
- DPO/: scaffolding for preference training (to be implemented)

Prerequisites
- Python 3.10+
- Recommended packages (install what you need per stage):
	- torch, tiktoken, numpy, pandas, tqdm, colorama, pydantic
	- python-dotenv, openai
	- youtube-transcript-api, google-api-python-client, isodate
	- docling (document_converter, chunking)

Environment variables (create a .env in repo root when needed)
- OPENAI_API_KEY for SFT data generation and verification
- YOUTUBE_API_KEY for fetching long-form video links

Hardware notes
- PoC runs on CPU/MPS/single GPU.
- GPT‑2 774M‑scale and long-context continued pretraining require strong multi‑GPU resources.

Evaluation
- Cross-entropy loss on held-out data for training sanity checks.
- Qualitative review (human/AI) for style alignment and instruction following.

Roadmap overview
1) PoC pretraining (scalable to GPT‑2 774M)
2) Continued pretraining on domain text (long‑form transcripts)
3) Supervised finetuning (instruction tuning)
4) DPO on a custom preference dataset (planned)

---

## 1) Proof of Concept: Pretraining (scalable to GPT‑2 774M)

Code
- Minimal GPT implementation: `pretraining/src/gpt.py`
- Config and device routing: `pretraining/src/config.py`
- Sampling utility: `pretraining/src/sample.py`
- Example notebook: `pretraining/src/training.ipynb`

Data (tiny Shakespeare for sanity checks)
- Helper: `pretraining/data/shakespeare/prepare.py` (downloads a tiny corpus and shows tiktoken BPE usage)

Notes
- The PoC model in code is a compact GPT that’s easy to run locally. It tokenizes by simple character set in sampling for simplicity; BPE tokenization is demonstrated in `prepare.py` if you want to align with GPT‑2’s 50257 vocabulary.
- To explore GPT‑2 774M‑like settings, increase config to approximately: n_layer≈36, n_head≈20, n_embd≈1280, block_size≈1024, vocab_size=50257 (requires BPE and significant compute). This repo keeps the training loop simple for learning; scaling to 774M should be done with distributed training infra.

Quick start (sampling)
1) Ensure `pretraining/src/config.py` points `data_path` to your text file and `model_save_path` to where checkpoints should live.
2) Generate text from the saved or untrained model:

	 - Script: `pretraining/src/sample.py` with args `--prompt` and `--max_tokens`

Outputs
- Checkpoints path: set via `GPTConfig.model_save_path`.
- Sampling prints generated text; optionally saves with `--output`.

---

## 2) Continued Pretraining on Long‑form Transcripts

Goal
- Further adapt the base model to Andrew Tate’s long‑form speaking style using transcripts from YouTube.

Data collection
- Fetch candidate long videos: `continued_pretraining/data/fetch_links.py`
	- Requires `YOUTUBE_API_KEY` in `.env`.
	- Saves a CSV (e.g., `fetched.csv`) with filtered long videos.
- Extract transcripts via API: `continued_pretraining/data/extract_api.py`
	- Reads a CSV of URLs (see `file_path` inside the script).
	- Writes plain‑text transcripts into `tate_long_form_data/`.
	- Skips files that already exist; reports successes/errors.

Training
- Notebook: `continued_pretraining/src/topgpt-continued-pre-training.ipynb`
	- Load transcripts, tokenize, and run continued pretraining against the PoC model (or a compatible initialization).

Tips
- Ensure consistent tokenization across stages if you switch from char-level to BPE.
- Monitor loss on a held-out slice for overfitting or drift.

---

## 3) Supervised Finetuning (Instruction Tuning)

Goal
- Create high-quality Q&A in Andrew Tate’s voice, verify quality automatically, and finetune an instruction‑following model.

Data generation
- `supervised_finetuning/data_processing/generate.py`
	- Converts a source PDF (default: `supervised_finetuning/data/andrew_tate_sft.pdf`) into chunks with Docling’s `HybridChunker`.
	- Calls the OpenAI API (set `OPENAI_API_KEY`) to synthesize Q&A pairs using `supervised_finetuning/data_processing/prompt.py`.
	- Streams JSON to console for transparency and saves `data/raw_data.json`.

Preprocess to flat instructions
- `supervised_finetuning/data_processing/preprocessing.py`
	- Flattens `raw_data.json` into `data/instructions.json` with `{"question", "answer"}` items.

Automated quality verification
- `supervised_finetuning/data_processing/verify.py`
	- Uses a verification prompt to score ACCURACY and STYLE (1–10 each).
	- Accepts items scoring >6 on both; appends accepted items to `data/verified_instructions.jsonl` and logs all scores to `data/verification_analysis.jsonl`.
	- Has a `start_idx` guard to resume mid‑dataset (default 500 in code—adjust as needed).

Training & inference notebooks
- `supervised_finetuning/src/gpt2_sft.ipynb`, `supervised_finetuning/src/draft_sft.ipynb`
- Inference: `supervised_finetuning/src/inference.ipynb`

Tips
- Keep prompts deterministic where possible during data generation for reproducibility.
- Periodically sample accepted/rejected examples to sanity‑check the verifier’s thresholds.

---

## 4) Planned: Direct Preference Optimization (DPO)

Goal
- Align responses with human preferences (or curator preferences) using pairs of (chosen, rejected) responses per prompt.

Repo scaffolding
- Folder: `DPO/` with `data/` and `src/` prepared for implementation.

Expected dataset format (to prepare)
- JSONL with items like:
	- `{ "prompt": str, "chosen": str, "rejected": str }`
- Source candidates can be synthesized from SFT outputs by sampling multiple responses per prompt and curating preferences, or by pairing model vs. reference outputs.

Planned approach
- Implement a DPO trainer under `DPO/src/` (e.g., PyTorch/TRL‑style loop) that:
	- Loads an SFT‑initialized policy model.
	- Trains on preference pairs without explicit reward modeling.
	- Evaluates alignment on a held‑out preference set.

Deliverables (upcoming)
- Scripts/notebooks for dataset creation from SFT outputs.
- A training script (or notebook) for DPO in `DPO/src/`.
- A small evaluation harness for preference accuracy.

---

## Practical notes

Tokenization consistency
- The PoC code demonstrates both simple character‑level tokenization (in `sample.py`) and GPT‑2 BPE via `tiktoken` (in `prepare.py`). For smooth multi‑stage training, prefer a single tokenizer spec (usually GPT‑2 BPE, vocab_size=50257) across stages.

Devices and checkpoints
- Devices are auto‑selected in `GPTConfig.device` (MPS → CUDA → CPU). Override if needed.
- Set `GPTConfig.model_save_path` to control where checkpoints are written/read.

Ethical & content considerations
- Long‑form and generated content may include controversial opinions. Use this code responsibly and comply with applicable laws, platform terms, and safety guidelines.

License
- See `LICENSE` in this repository.

Contact
- Open an issue or PR to discuss improvements, training recipes, and the upcoming DPO stage.

