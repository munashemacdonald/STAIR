# STAIR for 10-K Filings

A standalone project that applies STAIR ([Kumar et al., IBM, arXiv:2609.03874](https://arxiv.org/abs/2609.03874)) to SEC 10-K filings. STAIR is a fine-tuned LLM that retrieves by picking the right section from a document's table of contents. The project tests whether it beats standard RAG retrieval on 10-Ks, then shows how its cited answers can become signals for investment research. Timeline: Nov 2–30, 2026, at about 40 hours a week.

## Where things are

- `docs/PLAN.md`: the plan as a checklist, with no dates, organised by the same six objectives as the objectives plan and the one-pager. It is the source of truth for scope decisions.
- `docs/OBJECTIVES_PLAN.md`: steps and expected results for each objective. `OBJECTIVES_PLAN.pdf` is generated from it by `docs/build_plan_pdf.py`.
- `docs/STAIR_Project_Overview.pdf`: one-page topic, goal and objectives.
- `config.yaml`: companies, held-out companies, fiscal years, models, training settings, seed.
- `README.md`: layout, setup and the status checklist.

## Decisions so far

- **Scope:** retrieval accuracy and building signals. No stock-price prediction or returns backtest; that is future work.
- **Only STAIR is trained (Nov 9).** The comparisons are untrained: BM25, embedding retrieval, an untuned LLM given the ToC, and an LLM given the whole 10-K. The paper's STAIR vs. DSI result is cited, not reproduced.
  - **Added Nov 20**, so the comparison includes a modern RAG pipeline that needs no training:
    - **Hybrid:** BM25 + NV-Embed-v2, merged with reciprocal rank fusion (k = 60).
    - **Hybrid + reranker:** `BAAI/bge-reranker-v2-m3` over the hybrid's top 20 sections.
  - **The row to beat** is hybrid + reranker. Report the hybrid's R@20 as the reranker's ceiling.
  - **Say in the write-up** that STAIR is still the only method trained on these filings. A reranker fine-tuned on the same questions is future work.
- **Companies (Nov 17):** Technology sector.
  - Training companies, in pairs: chips (NVDA, AVGO), internet (GOOGL, META), software (CRM, ADBE), hardware (AAPL, DELL).
  - Held out: CRWV and FIG. Both listed in 2025 and have only an FY2025 10-K, so the held-out test is about first-time filers. They were chosen by filing availability, before any results existed.
  - ORCL was dropped to make room for FIG.
  - **Test-only (Nov 23):** ORCL and AMD, established companies that are never trained on and are tested on their FY2025 10-K only. One is software and one is chips. They separate "a company the model never saw" from "a company's first 10-K". Held-out results are reported for both groups (`group` field: `first_time` or `established`).
- **Fiscal years:** each company's own FY2024 and FY2025, so "FY2025" ends anywhere from Jan to Dec 2025 depending on the company. That makes 20 filings in total: 8 companies × 2 years, plus FY2025 for the 4 held-out and test-only companies.
- **Section targets** are numbered by position plus title, e.g. "7.3 Liquidity and Capital Resources", so every target is unique.
- **Hand-written test set (Nov 20): the headline result.** Every company's FY2025 filing, held-out and test-only companies included (12 companies), is asked up to about 450 questions, with every correct section labelled:
  - FinanceBench's 25 generic analyst questions (dg01–dg25, in `data/questions/financebench_templates.jsonl`)
  - 8 checklist questions (ck01–ck08, in `data/questions/checklist_templates.jsonl`), fixed before any table of contents was parsed
  - 5 company-specific questions per company, written before opening that company's table of contents
  - **Licence:** FinanceBench's templates are CC BY-NC 4.0, so credit FinanceBench and keep anything built from them non-commercial.
  - **Independent units:** a template's company versions move together, so the set has about 93 independent units (33 templates + 60 questions), not 450.
  - **Second labeller:** a classmate labels about 30 questions blind before the rest are labelled, and the agreement is reported.
  - **Kept out of training:** these questions and templates never appear in the question generator's prompt. Generated questions that closely match them are dropped.
  - FinanceBench's own questions can't train STAIR: only 150 are public, and they're about other companies and years.
  - When citing FinanceBench, its "81% wrong or refused" is GPT-4-Turbo with one search index shared by all filings. With one index per filing, which is closer to this project, it answered 50% correctly. FinanceBench scores answers, not retrieval.
- **Several correct sections (Nov 20):** 10-Ks repeat facts across Items, so a question can have more than one correct section.
  - **Labels:** each question has `answers_alone` (each answers by itself) and `needed_together` (they answer only together). The rule and line format are in `data/questions/README.md`.
  - **Generated questions:** dev and test questions get extra correct sections from an LLM judge, which checks the top 3 other sections from BM25 and embeddings (objectives plan, step 1.8).
  - **Pooling:** that check alone would favour BM25 and embeddings. So after every method runs, each method's top 3 sections are judged without knowing which method found them, and the yeses are recorded in `pooled_answers_alone` (step 3.6). Results are reported with and without them.
  - **Scoring:** R@k means any correct section (`answers_alone` or `pooled_answers_alone`) is in the top k (the official code's `match@k`), not its `recall@k`. Needed-together questions are reported separately, with All@3.
  - **Confidence intervals** for the hand-written set resample templates and company-specific questions as units, not single questions. The headline is the CI of STAIR's difference from each method.
- **Statistics (Nov 23; objectives plan, step 3.7):**
  - **Primary comparison,** fixed in `config.yaml` (`scoring.primary`) before any test: STAIR (official) against hybrid + reranker, R@1 on the hand-written set, judged by the CI of the difference.
  - **Every other comparison is secondary,** with Holm-adjusted p-values.
  - **Other test sets:** synthetic and held-out sets resample sections. The held-out groups get per-company results and no significance claims.
  - **Small cells:** under about 30 questions, show the count and CI, and draw no conclusion from them.
  - **Freeze the test sets:** every choice is made on dev. Record the first test run's date.
  - **Same subset:** every method is also scored on the whole-10-K method's 300-question synthetic sample (a like-for-like table).
- **API models may have seen these filings (Nov 23):** record each API model's training-data cutoff. Break down its results by filings published before and after the cutoff, and state it as a limitation. Mistral-7B-v0.2 predates every filing.
- **Objective 5 (Nov 23):**
  - **Two pipelines:** the checklist runs twice, with STAIR's top 3 sections and with hybrid + reranker's, using the same answering model.
  - **Independent checks:** scores are checked against your own scores and, where the WRDS subscription has the data, Audit Analytics (material weaknesses) and Compustat (customers, debt maturities DD1–DD5, buybacks PRSTKC). WRDS data is never committed.
  - **Cited-answer check:** about 50 hand-written questions are judged for correct, supported and a valid citation (step 5.6).
- **Licences (checked Nov 23; table in the objectives plan):**
  - **Apache 2.0:** Mistral-7B-Instruct-v0.2, Qwen2.5-14B-Instruct and bge-reranker-v2-m3.
  - **Non-commercial (CC BY-NC 4.0):** NV-Embed-v2 and the FinanceBench templates, so the project is non-commercial.
  - **API models:** their outputs are baselines or judges only. They never become STAIR training data unless the model's licence allows it.
- **Official code (Nov 19):** v2 of the paper links the authors' code and data, and both are public. The v1 link is dead.
  - Code: [github.com/meghanadhpulivarthi/stair-clean](https://github.com/meghanadhpulivarthi/stair-clean), commit `3729c54`.
  - Data: Hugging Face `meghanadh/SearchTome`, CC BY-SA 4.0. It holds one textbook per domain, while the paper used 18 books.
  - The project builds on this code and adapts it for 10-Ks; see "Adapting the official code" below.
- **Where models run (Nov 19; moved to Colab Nov 24):** see `runs_on` and `machines` in `config.yaml`.
  - **On a Colab GPU,** paid through the user's Google AI Pro plan (about 200 compute units a month):
    - question writing and every LLM judge (Qwen2.5-14B, 4-bit)
    - STAIR training and inference (the title trie and top-3 beam search need local generation)
    - NV-Embed-v2 embeddings and the reranker (run after NV-Embed-v2 is unloaded)
    - the untuned Mistral baseline (same model and GPU type as STAIR, for a fair comparison)
    - objective 5's answering model
  - **On the laptop (CPU):** downloading and parsing, the hand-written questions and their labels, BM25, merging the hybrid's rankings, the API baselines, scoring, statistics and the write-up.
  - **Through Hugging Face Inference Providers:** the stronger-LLM baseline (70B-class) and the whole-10-K baseline.
  - **The school GPU is the fallback.** Its Windows setup stays in the objectives plan's appendix.
- **Laptop–Colab workflow (Nov 24):**
  - **Every step is a script** in `src/` that reads files and writes files, so nothing depends on a notebook's memory and no notebook ever switches kernels.
  - **Files move between the two** through GitHub (code) and Google Drive (data, checkpoints, run outputs). The Drive desktop app syncs `My Drive/stair/` to the laptop.
  - **One Colab notebook,** `notebooks/colab_gpu.ipynb`, run in the browser: its cells mount Drive, read tokens from Colab Secrets, pull the repos and start one script. Commit it without outputs.
  - **About three GPU sessions:** question writing and judging; training; then every GPU method on every test set.
  - **The VS Code Colab extension** is only for short GPU jobs. Colab Secrets don't work there (enter tokens with `getpass`), and it's untested on long runs.

## Adapting the official code

**Terms and record-keeping**
- In the write-up, call it the "official implementation" or "starting codebase", not a "baseline". "Baseline" means the comparison methods.
- Record the commit, and list every change you make.

**Licence**
- The repo has no licence file. Keep it in a separate private copy, and don't copy its code into this repo until the authors give permission (requested in the Nov 19 email). Don't use a GitHub fork: a fork of a public repo is always public, so pushing changes there would publish them.
- Keep our changes as a separate module or patch, so this repo holds only our own code whatever the authors decide.

**What already fits**
- **Table-of-contents format:** `toc.json` is `{"title": ..., "table_of_contents": [{"id", "section_num", "title"}]}`. Our parser writes this directly.
- **Output format:** the model outputs a list of `"section_num title"`, the same as our numbered targets.
- **Question generation:** it calls any OpenAI-compatible endpoint, so it can run through the Hugging Face API.
- **Metrics:** its eval reports Recall, nDCG and MRR at 1/3/5/10, and counts invalid titles.

**Changes needed for 10-Ks**
- **Many filings per run:** the code reads one table of contents per dataset. Each question must carry its filing's id and be paired with that filing's ToC, for both training and evaluation.
- **Prompt:** it says "Book: {title}". Name the company and filing instead, e.g. "NVIDIA Corp., Form 10-K, FY2025".
- **Input:** skip its PDF `prepare-data` step. Our 10-K parser writes the ToC and section text.
- **4-bit loading:** the code loads models in bf16 only, which is too tight for Mistral-7B on an L4's 24 GB once training overhead is added. Add 4-bit QLoRA (`bitsandbytes`), and use it on every GPU, A100 included, so runs match whichever GPU Colab assigns.
- **Length:** `max_seq_length` is 2,048. Measure the longest ToC plus question, and raise it.
- **Checkpoint choice:** the code keeps the lowest dev loss. The paper chose by dev Recall@1; switch if there's time, and say which one was used.
- **Restricted output:** its eval generates freely (greedy) and counts invalid titles afterwards. Keep that as the faithful replication. The trie is our extension, so report both.
  - **Trie format:** build it over the full list strings the model is trained to output, `["<section_num> <title>"]`, not bare titles, which would block the opening `["`. Tokenize them exactly like the training targets.
  - **Top 3:** the official version's top 3 is the first 3 valid items of its generated list, and is often just one item. The restricted version's top 3 is its 3 best beams out of 5.
- **Training budget:** at most about 8 hours per full run (`training.max_hours_per_run`). Estimate it first from a 200-step timing: questions × prompt tokens ÷ tokens per second.
  - If it's over budget, cut in this order: cap the questions per section (e.g. 20), use a two-level ToC (relabel questions to the level-2 parent), train fewer epochs, then a smaller model as a last resort.
  - Record the estimate and the actual time.
- **Save often and make training resumable:**
  - Save a checkpoint about every 30 minutes to Google Drive (`/content/drive/MyDrive/stair/checkpoints/`), never to `/content`, which Colab wipes. Set `save_interval` from the timing; `eval_interval` must equal it or divide it.
  - Add a `--resume` option to your copy's `train` command that passes `resume_from_checkpoint=True`, and test it once.
  - After an interruption, resume instead of restarting.

**Running on Colab** (the full steps are in `docs/OBJECTIVES_PLAN.md`, "Setting up the laptop and Colab")
- **Colab is Linux,** so the official code runs as the authors wrote it: `torchrun` works, and none of the Windows changes below are needed there. Only `bitsandbytes` is added to your copy.
- **Don't reinstall torch.** Colab comes with a CUDA build; `pip install -r requirements.txt` keeps it.
- **Check the GPU before any training:** `torch.cuda.is_available()` must be `True`, and the device name must be the L4 or A100 you chose. Record the GPU type and Colab's Python version for every run.
- **Keep your copy private** (the official code has no licence). Make it with GitHub's "Import repository", not a fork. Colab clones it with a read-only GitHub token kept in Colab Secrets.
- **Sessions end and wipe `/content`.** Google AI Pro has no background execution, so keep the tab open and the laptop awake during a run. Write everything worth keeping to Drive, and resume from checkpoints after a disconnect.
- **Models download from Hugging Face in each session,** to Colab's disk. Don't store them on Drive.
- **When a session's work is done,** use Runtime → Disconnect and delete runtime, because units burn while a runtime is connected. Record the units used.
- **Generation stays on `transformers`** (4-bit, batched). vLLM works on Colab and could speed up question writing if needed, but STAIR's restricted output needs `transformers`' `prefix_allowed_tokens_fn`.

**Fallback: the school GPU under Windows** (full steps in the objectives plan's appendix). Use Python 3.15.2 with `uv` installed for the user only. Install torch from PyTorch's CUDA 13.0 index (`https://download.pytorch.org/whl/cu130`; a plain install gets the CPU build), and check that the device is the RTX 4000 SFF Ada. In your copy of the official code:
- start the trainer with `sys.executable` instead of `torchrun`, because Windows lacks NCCL and torchrun's libuv rendezvous
- set `PYTHONUTF8=1`
- point `HF_HOME` at a folder with about 60 GB free
- don't use `torch.compile` (no Triton on Windows)

`ranx` won't install there, because `lz4` and `fastparquet` have no Windows builds for 3.15 (Nov 24), so score on the laptop.

**Order of work**
1. Run the bundled example.
2. Reproduce one SearchTome domain (finance) and compare with the paper.
3. Make the 10-K changes one at a time, running the repo's tests after each.

## Environment

- **Laptop:** Windows 11, no NVIDIA GPU. It runs every step that needs no GPU, in `.venv`.
  - **Smart App Control is on.** It blocks unsigned programs, and uv's own Python builds are unsigned: a uv-downloaded 3.15.2 failed to load. So `.venv` uses the signed python.org **Python 3.14.0** (built Nov 24), with CPU-only torch.
  - Always give uv the python.org interpreter's path, and never let it download a Python. Never suggest turning Smart App Control off: it can't be turned back on without reinstalling Windows.
  - Python 3.15 would also fail here, because `ranx`'s `lz4` and `fastparquet` have no Windows builds for 3.15 yet (Nov 24).
  - Importing `torch` and `transformers` takes about 35 seconds on this laptop.
- **Colab (GPU):** paid through the user's Google AI Pro plan, about 200 compute units a month. The whole project is estimated at about 100.
  - **L4** (24 GB, bf16, about 1.7 units an hour): about the school GPU's speed. Use it for question writing, embeddings, the reranker and inference.
  - **A100** (40 GB, about 5.4 units an hour; the 80 GB version about 7.5): about 4–5× faster. Use it for training when Colab offers it; it isn't always available.
  - **T4** (16 GB, free): no bf16, and NV-Embed-v2 doesn't fit. Quick tests only.
  - These rates are third-party measurements from March 2026; check the balance in Colab.
  - Claude Code runs on the laptop. It can edit code and notebooks but can't run anything on Colab.
- **What fits on an L4 (24 GB):**
  - QLoRA on Mistral-7B, with batch size 1 and gradient checkpointing
  - 7–14B models in 4-bit, for writing questions
  - NV-Embed-v2, for inference only
- **What doesn't fit on an L4:**
  - Mixtral 8x7B
  - 70B-class models, and the whole-10-K baseline's long context: both go through the Hugging Face API
- **Speed:** time a short run before planning long ones. See the training budget above.
- **Fallback GPU:** a school computer with an NVIDIA RTX 4000 SFF Ada (20 GB VRAM, 70 W), driver 596.71, Windows, no admin rights, Python 3.15.2. Its Intel UHD 770 is the built-in graphics chip and isn't used.

## Rules

- Never commit anything under `data/raw/`, `data/parsed/`, `checkpoints/` or `papers/`; they are gitignored. Do commit `data/questions/`, especially the hand-written questions, which can't be regenerated.
- **SEC EDGAR:**
  - Every request needs a User-Agent containing a name and email, and must stay under 10 requests per second.
  - Read the identity from the `SEC_IDENTITY` environment variable. Don't commit a real email.
- **Hugging Face:**
  - Read the token from the `HF_TOKEN` environment variable. Never put it in code, `config.yaml` or any committed file.
  - Mistral-7B-Instruct-v0.2 is not gated (checked Nov 23), so it downloads with the token alone.
  - **API calls:**
    - Save every response under `data/api_cache/` (gitignored) and reuse it, so reruns don't cost money twice.
    - Use temperature 0.
    - Estimate the token cost before a large run; the whole-10-K baseline is the expensive one.
    - Record the provider, model and date for the write-up.
- **Colab and Google Drive:**
  - Keep tokens out of notebook cells and files. Use Colab Secrets in the browser, or `getpass` in the VS Code extension.
  - Commit notebooks with their outputs cleared.
  - Never upload WRDS data to Colab or Drive.
  - Drive holds parsed filings, generated questions, checkpoints and run outputs. Generated questions are also copied into `data/questions/` and committed.
- When `docs/OBJECTIVES_PLAN.md` changes, regenerate its PDF with `.venv\Scripts\python.exe docs\build_plan_pdf.py`, or say that the PDF is out of date.
- The user is a financial engineering student. Explain technical terms the first time they come up, and ask before committing or pushing.
