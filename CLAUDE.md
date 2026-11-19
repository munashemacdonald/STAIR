# STAIR for 10-K Filings

A standalone project that applies STAIR ([Kumar et al., IBM, arXiv:2609.03874](https://arxiv.org/abs/2609.03874)) to SEC 10-K filings. STAIR is a fine-tuned LLM that retrieves by picking the right section from a document's table of contents. The project tests whether it beats standard RAG retrieval on 10-Ks, then shows how its cited answers can become signals for investment research. Timeline: Nov 2–30, 2026, at about 40 hours a week.

## Where things are

- `docs/PLAN.md`: the dated plan. It is the source of truth for scope decisions.
- `docs/OBJECTIVES_PLAN.md`: steps and expected results for each objective. `OBJECTIVES_PLAN.pdf` is generated from it.
- `docs/STAIR_Project_Overview.pdf`: one-page topic, goal and objectives.
- `config.yaml`: companies, held-out companies, fiscal years, models, training settings, seed.
- `README.md`: layout, setup and the status checklist.

## Decisions so far

- **Scope:** retrieval accuracy and building signals. No stock-price prediction or returns backtest; that is future work.
- **Only STAIR is trained (Nov 9).** The comparisons are untrained: BM25, embedding retrieval, an untuned LLM given the ToC, and an LLM given the whole 10-K. The paper's STAIR vs. DSI result is cited, not reproduced.
- **Companies (Nov 17):** Technology sector.
  - Training companies, in pairs: chips (NVDA, AVGO), internet (GOOGL, META), software (CRM, ADBE), hardware (AAPL, DELL).
  - Held out: CRWV and FIG. Both listed in 2025 and have only an FY2025 10-K, so the held-out test is about first-time filers. They were chosen by filing availability, before any results existed.
  - ORCL was dropped to make room for FIG.
- **Fiscal years:** each company's own FY2024 and FY2025, so "FY2025" ends anywhere from Jan to Dec 2025 depending on the company. That makes 18 filings in total.
- **Section targets** are numbered by position plus title, e.g. "7.3 Liquidity and Capital Resources", so every target is unique.
- **Official code (Nov 19):** v2 of the paper links the authors' code and data, and both are public. The v1 link is dead.
  - Code: [github.com/meghanadhpulivarthi/stair-clean](https://github.com/meghanadhpulivarthi/stair-clean), commit `3729c54`.
  - Data: Hugging Face `meghanadh/SearchTome`, CC BY-SA 4.0. It holds one textbook per domain, while the paper used 18 books.
  - The project builds on this code and adapts it for 10-Ks; see "Adapting the official code" below.
- **Where models run (Nov 19):** see `runs_on` in `config.yaml`.
  - **On the school GPU:** STAIR training and inference (the title trie and top-3 beam search need local generation), BM25, embeddings, and the untuned Mistral baseline (same model and machine as STAIR, for a fair comparison).
  - **Through Hugging Face Inference Providers:** the stronger-LLM baseline (70B-class) and the whole-10-K baseline. Question writing can move there to free the GPU.

## Adapting the official code

**Terms and record-keeping**
- In the write-up, call it the "official implementation" or "starting codebase", not a "baseline". "Baseline" means the comparison methods.
- Record the commit, and list every change you make.

**Licence**
- The repo has no licence file. Keep it in a separate fork or checkout, and don't copy its code into this repo until the authors give permission (requested in the Nov 19 email).

**What already fits**
- **Table-of-contents format:** `toc.json` is `{"title": ..., "table_of_contents": [{"id", "section_num", "title"}]}`. Our parser writes this directly.
- **Output format:** the model outputs a list of `"section_num title"`, the same as our numbered targets.
- **Question generation:** it calls any OpenAI-compatible endpoint, so it can run through the Hugging Face API.
- **Metrics:** its eval reports Recall, nDCG and MRR at 1/3/5/10, and counts invalid titles.

**Changes needed for 10-Ks**
- **Many filings per run:** the code reads one table of contents per dataset. Each question must carry its filing's id and be paired with that filing's ToC, for both training and evaluation.
- **Prompt:** it says "Book: {title}". Name the company and filing instead, e.g. "NVIDIA Corp., Form 10-K, FY2025".
- **Input:** skip its PDF `prepare-data` step. Our 10-K parser writes the ToC and section text.
- **4-bit loading:** the code loads models in bf16 only, which is too tight for Mistral-7B in 20 GB. Add 4-bit QLoRA (`bitsandbytes`).
- **Length:** `max_seq_length` is 2,048. Measure the longest ToC plus question, and raise it.
- **Checkpoint choice:** the code keeps the lowest dev loss. The paper chose by dev Recall@1; switch if there's time, and say which one was used.
- **Restricted output:** its eval generates freely (greedy) and counts invalid titles afterwards. Keep that as the faithful replication. The title trie is our extension, so report both.

**Running on the GPU under Windows** (the full steps are in `docs/OBJECTIVES_PLAN.md`, "Before you start")
- **Python and environments:** use the installed Python 3.15.2, with `uv` installed for the user only (no admin needed). PyTorch and every compiled dependency have Windows builds for 3.15 (checked Nov 19).
- **PyTorch's CUDA build:** a plain `pip`/`uv` install of torch on Windows gets the CPU-only build. Always install from PyTorch's CUDA 13.0 index, `https://download.pytorch.org/whl/cu130`, which gives torch 2.14.1+cu130.
  - In the fork, set this through `[[tool.uv.index]]` and `[tool.uv.sources]` in `pyproject.toml`.
  - In this repo, use `--index-url`.
  - No CUDA toolkit install is needed; the driver (596.71) is enough.
- **Check the GPU before any training:** `torch.cuda.is_available()` must be `True`, and the device name must be the RTX 4000 SFF Ada.
- **No `torchrun`:** in the fork's `src/stair/train.py`, start the trainer with `sys.executable` instead of `torchrun` when `num_gpus` is 1. One GPU needs no distributed setup, and this avoids two things Windows lacks: NCCL and torchrun's libuv rendezvous.
- **Add `bitsandbytes`** to the fork's dependencies for 4-bit loading. It ships Windows builds.
- **Text encoding:** the official code opens files without an encoding. Python 3.15 defaults to UTF-8; also set `PYTHONUTF8=1` as a safeguard.
- **Disk space:** set `HF_HOME` to a folder with about 60 GB free (Mistral-7B, Qwen2.5-14B, NV-Embed-v2). Without admin rights, Hugging Face stores full copies on Windows instead of links.
- **No `torch.compile`:** it needs Triton, which isn't available on Windows. Attention uses PyTorch's built-in SDPA.

**Order of work**
1. Run the bundled example.
2. Reproduce one SearchTome domain (finance) and compare with the paper.
3. Make the 10-K changes one at a time, running the repo's tests after each.

## Environment

- **GPU machine:** a separate computer with an NVIDIA RTX 4000 SFF Ada (20 GB VRAM, 70 W), driver 596.71.
  - It runs Windows and is a school computer, so Linux/WSL can't be installed. Everything runs natively on Windows (Python 3.15.2 and `uv`, user-level installs). No vLLM: generate with Hugging Face `transformers` (4-bit, batched), or Ollama if that's too slow. No FlashAttention; use PyTorch's built-in SDPA.
  - Its Intel UHD 770 is the built-in graphics chip and isn't used.
- **No GPU elsewhere:** the user's laptop has no NVIDIA GPU, and cloud sessions have no GPU either.
- **What fits in 20 GB:**
  - QLoRA on Mistral-7B, with batch size 1 and gradient checkpointing
  - 7–14B models in 4-bit, for writing questions
  - NV-Embed-v2, for inference only
- **What doesn't fit:**
  - Mixtral 8x7B
  - 70B-class models, and the whole-10-K baseline's long context: both go through the Hugging Face API
- **Speed:** it's a low-power card, so time a short run before planning long ones.

## Rules

- Never commit anything under `data/raw/`, `data/parsed/`, `checkpoints/` or `papers/`; they are gitignored. Do commit `data/questions/`, especially the hand-written questions, which can't be regenerated.
- **SEC EDGAR:**
  - Every request needs a User-Agent containing a name and email, and must stay under 10 requests per second.
  - Read the identity from the `SEC_IDENTITY` environment variable. Don't commit a real email.
- **Hugging Face:**
  - Read the token from the `HF_TOKEN` environment variable. Never put it in code, `config.yaml` or any committed file.
  - Mistral-7B-Instruct is gated: accept its terms on the model page before the first download.
  - **API calls:**
    - Save every response under `data/api_cache/` (gitignored) and reuse it, so reruns don't cost money twice.
    - Use temperature 0.
    - Estimate the token cost before a large run; the whole-10-K baseline is the expensive one.
    - Record the provider, model and date for the write-up.
- When `docs/OBJECTIVES_PLAN.md` changes, regenerate its PDF or say that the PDF is out of date.
- The user is a financial engineering student. Explain technical terms the first time they come up, and ask before committing or pushing.
