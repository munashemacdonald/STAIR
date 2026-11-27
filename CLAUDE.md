# STAIR for 10-K Filings

A standalone project that applies STAIR ([Kumar et al., IBM, arXiv:2609.03874](https://arxiv.org/abs/2609.03874)) to SEC 10-K filings. STAIR is a fine-tuned LLM that retrieves by picking the right section from a document's table of contents. The project tests whether it beats standard RAG retrieval on 10-Ks, then shows how its cited answers can become signals for investment research. Timeline: started Nov 2, 2026, at about 40 hours a week. On Nov 25 about 15 days were left, and the plan was cut to the core test (see "What's in, and what's left out" in the objectives plan).

**The core test (Nov 25):** does STAIR, trained on FY2024 tables of contents, find the right section of an FY2025 10-K more often than the best standard RAG pipeline (hybrid + reranker)? It's measured at two levels:
- **temporal zero-shot** (`seen_company`): seen companies in a new year
- **cross-company zero-shot** (`first_time` + `established`): companies it never saw

This is the question the paper leaves open in its Section 7: the paper trains and tests on the same books, so its 82.6% isn't comparable. Either outcome is a finding.

**Keep it simple.** The user asked on Nov 25 for one version of STAIR and no nice-to-have extras. Anything outside the core test is future work. Don't add methods, variants or analyses without asking.

## Where things are

- `docs/PLAN.md`: the plan as a checklist, organised by the same objectives as the objectives plan and the overview. It is the source of truth for scope decisions.
- `docs/OBJECTIVES_PLAN.md`: steps and expected results for each objective, and the list of what was left out. `OBJECTIVES_PLAN.pdf` is generated from it by `docs/build_pdfs.py`.
- `docs/STAIR_Project_Overview.pdf`: the overview (topic, goal and objectives; two pages). It is generated from `docs/STAIR_Project_Overview.html` by the same script.
- `config.yaml`: companies, test groups, fiscal years, models, prompt, question settings, scoring, training settings, seed.
- `README.md`: layout, setup and the status checklist.

## Decisions so far

- **Scope:** retrieval accuracy and building signals. No stock-price prediction or returns backtest.
- **One version of STAIR (Nov 25):** 5-beam search with no constraint on its output. R@1 uses the top beam; the top 3 are the valid sections from the beams. Invalid titles are counted.
- **Only STAIR is trained (Nov 9).** The five comparison methods are untrained:
  - BM25, embeddings (NV-Embed-v2), and their hybrid by reciprocal rank fusion (k = 60)
  - **hybrid + reranker** (`BAAI/bge-reranker-v2-m3` over the hybrid's top 20): the row to beat
  - the untuned Mistral given the table of contents, with STAIR's prompt and decoding
  - Say in the write-up that STAIR is the only method trained on these companies' filings.
- **Chunks (Nov 25):** every RAG method works on ~300-word chunks overlapping by 50 words, each starting with its section number and title. A section scores as its best chunk (MaxP).
- **Companies (Nov 17):** Technology sector.
  - Training companies, in pairs: chips (NVDA, AVGO), internet (GOOGL, META), software (CRM, ADBE), hardware (AAPL, DELL).
  - Held out: CRWV and FIG, which listed in 2025 and have only an FY2025 10-K (first-time filers). ORCL was dropped to make room for FIG.
  - **Test-only (Nov 23):** ORCL and AMD, established companies never trained on, tested on FY2025 only.
- **Fiscal years:** each company's own FY2024 and FY2025, so "FY2025" ends anywhere from Jan to Dec 2025. That makes 20 filings: 8 companies × 2 years, plus FY2025 for the 4 unseen companies.
- **Train on FY2024, test on FY2025 (Nov 25):**
  - STAIR trains on the 8 training companies' FY2024 filings only. Every test set uses FY2025 filings, so no tested filing was ever trained on.
  - Dev is 20% of the 8 companies' FY2025 sections, and `test_synthetic` the other 80%, split by section. The 4 unseen companies' questions are `test_heldout`.
  - Each FY2025 question has a `group`: `seen_company`, `first_time` or `established`.
- **No company names (Nov 25):** the prompt header is "SEC Form 10-K filing", with no company or year, and every question says "the company", for every method. A table of contents can still reveal the company; say so in the write-up.
- **Generated questions (Nov 25):**
  - Written per section: 3–6 per FY2024 section for training, about 5,000 in all, and 2 per FY2025 section for dev and test.
  - Generation, the pooling judge and objective 5's answers run on Colab with vLLM (`Qwen/Qwen2.5-14B-Instruct-AWQ`, Apache 2.0).
  - Every training target is exactly one section.
  - Filtered for duplicates and answerability, and 100 are read by hand.
- **Section targets** are numbered by position plus title, e.g. "7.3 Liquidity and Capital Resources", so every target is unique.
- **Hand-written test set: the headline.** Every company's FY2025 filing (12 companies) is asked about 200 questions, with every correct section labelled:
  - 7 FinanceBench templates: dg09, dg11, dg20 and dg22, which objective 5 uses, plus dg14, dg19 and dg24. They were chosen before any table of contents was parsed (`handwritten` in `config.yaml`).
  - 8 checklist templates (ck01–ck08), fixed before any table of contents was parsed
  - 36 company-specific questions: 2 per seen company and 5 per unseen company, all written before building the parser
  - **Licence:** FinanceBench's templates are CC BY-NC 4.0; credit FinanceBench and keep anything built from them non-commercial.
  - **Independent units:** a template's company versions move together. The seen group has about 31 units and the unseen group about 35, so only differences of roughly 20 points or more are likely to reach significance within a group.
  - **Second labeller:** a classmate labels about 30 questions blind before the rest are labelled, and the agreement is reported.
  - **Kept out of training:** these questions and templates never appear in the question generator's prompt, and generated questions that closely match them are dropped.
  - When citing FinanceBench: its "81% wrong or refused" is GPT-4-Turbo with one search index shared by all filings. With one index per filing, closer to this project, it answered 50% correctly. FinanceBench scores answers, not retrieval.
- **Several correct sections:** 10-Ks repeat facts across Items, so a question can have more than one correct section.
  - **Labels:** `answers_alone` (each answers by itself) and `needed_together` (they answer only together). The rule and line format are in `data/questions/README.md`.
  - **Pooling:** after every method runs, each method's top 3 sections are judged blind by the LLM judge, and the yeses go in `pooled_answers_alone` (step 3.7). For the hand-written set, 50 of its decisions are checked by hand. Results are reported with and without pooling.
  - **Scoring:** R@k means any correct section is in the top k (the official code's `match@k`). Needed-together questions are left out of R@k and reported in one line with All@3.
- **Statistics (objectives plan, step 3.8):**
  - **Two co-primary comparisons,** fixed in `config.yaml` (`scoring.primary`) before any test: STAIR against hybrid + reranker, R@1 on the hand-written set, for the seen companies and for the unseen companies. Each is judged by the bootstrap CI of the difference, resampling templates and company-specific questions as units, with Holm across the two.
  - **Every other comparison is secondary,** Holm-adjusted. The synthetic held-out set gets full tests, resampling sections, because it gives cross-company zero-shot its power; it covers only 4 companies.
  - **Small cells:** under about 30 questions, show the count and CI, and draw no conclusion.
  - **Freeze the test sets:** every choice is made on dev. Record the first test run's date.
- **Objective 4:** label each seen company's FY2025 section as unchanged, renamed or new, and break down accuracy by that label.
- **Objective 5:** the 12-question checklist is run for every company and year with STAIR's top 3 sections, each capped at about 3,000 tokens (best chunks by the reranker). The year-over-year change makes the signal table, and 20 answers are checked by hand.
- **Licences (checked Nov 23; table in the objectives plan):**
  - **Apache 2.0:** Mistral-7B-Instruct-v0.2, Qwen2.5-14B-Instruct (and its AWQ version) and bge-reranker-v2-m3.
  - **Non-commercial (CC BY-NC 4.0):** NV-Embed-v2 and the FinanceBench templates, so the project is non-commercial.
- **Official code (Nov 19):** v2 of the paper links the authors' code: [github.com/meghanadhpulivarthi/stair-clean](https://github.com/meghanadhpulivarthi/stair-clean), commit `3729c54`. The project builds on it; see "Adapting the official code" below.
- **Where models run (Nov 24):** see `runs_on` and `machines` in `config.yaml`. No paid API is used.
  - **On a Colab GPU,** paid through the user's Google AI Pro plan (about 200 compute units a month): question writing, the pooling judge, STAIR training and inference, NV-Embed-v2, the reranker, the untuned Mistral, and objective 5's answers.
  - **On the laptop (CPU):** downloading and parsing, the hand-written questions and labels, BM25, merging the hybrid's rankings, scoring, statistics, objective 4 and the write-up.
  - **The school GPU is the fallback.** Its Windows setup is in the objectives plan's appendix.
- **Laptop–Colab workflow (Nov 24):**
  - **Every step reads files and writes files,** so no notebook ever switches kernels.
  - **The user works best with notebooks and runs the code themself (Nov 26).** Laptop steps go in notebooks under `notebooks/`, one per objective (e.g. `notebooks/objective1_dataset.ipynb`), with the code in the cells and a markdown cell explaining each step. Claude writes the cells; the user runs them and saves the notebook so Claude can read the outputs. Shared code that Colab also needs goes in `src/`.
  - **Files move between the two** through GitHub (code) and Google Drive (data, checkpoints, run outputs). The Drive desktop app syncs `My Drive/stair/` to the laptop.
  - **One Colab notebook,** `notebooks/colab_gpu.ipynb`, run in the browser: its cells mount Drive, read tokens from Colab Secrets, pull the repos and start one script. Commit it without outputs.
  - **About three GPU sessions:** question writing; training; then every GPU method, the pooling judge and objective 5's answers.
  - **The VS Code Colab extension** is only for short GPU jobs. Colab Secrets don't work there (enter tokens with `getpass`).

## Adapting the official code

**Terms and record-keeping**
- In the write-up, call it the "official implementation" or "starting codebase", not a "baseline". "Baseline" means the comparison methods.
- Record the commit, and list every change you make.

**Licence**
- The repo has no licence file. Keep it in a separate private copy, made with GitHub's "Import repository", and don't copy its code into this repo until the authors give permission (requested in the Nov 19 email). Don't use a GitHub fork: a fork of a public repo is always public, so pushing changes there would publish them.
- Keep our changes as a separate module or patch, so this repo holds only our own code whatever the authors decide.

**What already fits**
- **Table-of-contents format:** `toc.json` is `{"title": ..., "table_of_contents": [{"id", "section_num", "title"}]}`. Our parser writes this directly, with the title "SEC Form 10-K filing".
- **Output format (checked in the code Nov 25):** the prompt asks for "a Python list of strings", and the training completion is a newline followed by the list, for example `\n["7.3 Liquidity and Capital Resources"]`. The paper's Table 2 shows a single string instead; we follow the code, and the write-up says so.
- **Metrics:** its eval reports Recall, nDCG and MRR at several cutoffs, and counts invalid titles.

**Changes needed for 10-Ks**
- **Many filings per run:** the code reads one table of contents per dataset. Each question must carry its filing's id and be paired with that filing's ToC, for both training and evaluation.
- **Prompt:** it says "Book: {title}". The title is "SEC Form 10-K filing", with no company name or year, and questions say "the company".
- **Input:** skip its PDF `prepare-data` step. Our 10-K parser writes the ToC and section text.
- **4-bit loading:** the code loads models in bf16 only, which is too tight for Mistral-7B on an L4's 24 GB once training overhead is added. Add 4-bit QLoRA (`bitsandbytes`), and use it on every GPU, A100 included.
- **Length:** `max_seq_length` is 2,048. Measure the longest ToC plus question, and raise it.
- **Decoding:** the code's eval decodes greedily, which returns one section. Use 5-beam search instead, with no constraint, so R@3 is meaningful (step 2.6).
- **Checkpoint choice:** choose by dev R@1, not the code's default, dev loss. Keep every checkpoint (`save_total_limit: null`), score each on the 300-question dev sample with the test decoding, and keep the best. Dev holds seen companies only, so this tunes for temporal zero-shot; say so in the write-up.
- **Training budget:** at most about 8 hours and **at most 5 epochs** per run. The paper's 200 epochs memorised each book; here that would memorise FY2024 and hurt FY2025. Estimate the time first from a 200-step timing: questions × prompt tokens ÷ tokens per second.
  - If it's over budget, cut in this order: a two-level ToC (relabel questions to the level-2 parent), fewer epochs, then a smaller model.
  - Record the estimate and the actual time.
- **Save often and make training resumable:**
  - Save a checkpoint about every 30 minutes to Google Drive (`/content/drive/MyDrive/stair/checkpoints/`), never to `/content`, which Colab wipes. Set `save_interval` from the timing; `eval_interval` must equal it or divide it.
  - Add a `--resume` option to your copy's `train` command that passes `resume_from_checkpoint=True`, and test it once.
  - After an interruption, resume instead of restarting.

**Running on Colab** (full steps in `docs/OBJECTIVES_PLAN.md`, "Setting up the laptop and Colab")
- **Colab is Linux,** so the official code runs as the authors wrote it: `torchrun` works, and none of the Windows changes below are needed. Only `bitsandbytes` is added to your copy.
- **Don't reinstall torch.** Colab comes with a CUDA build; `pip install -r requirements.txt` keeps it. vLLM installs its own torch, so use it only in a session that does nothing else.
- **Check the GPU before any training:** `torch.cuda.is_available()` must be `True`, and the device must be the L4 or A100 you chose. Record the GPU type and Colab's Python version for every run.
- **Keep your copy private.** Colab clones it with a read-only GitHub token kept in Colab Secrets.
- **Sessions end and wipe `/content`.** Google AI Pro has no background execution, so keep the tab open and the laptop awake during a run. Write everything worth keeping to Drive.
- **Models download from Hugging Face in each session,** to Colab's disk. Don't store them on Drive.
- **When a session's work is done,** use Runtime → Disconnect and delete runtime, because units burn while a runtime is connected. Record the units used.

**Fallback: the school GPU under Windows** (full steps in the objectives plan's appendix). Use Python 3.15.2 with `uv` installed for the user only. Install torch from PyTorch's CUDA 13.0 index (`https://download.pytorch.org/whl/cu130`; a plain install gets the CPU build), and check that the device is the RTX 4000 SFF Ada. In your copy of the official code:
- start the trainer with `sys.executable` instead of `torchrun`, because Windows lacks NCCL and torchrun's libuv rendezvous
- set `PYTHONUTF8=1`
- point `HF_HOME` at a folder with about 60 GB free
- don't use `torch.compile` (no Triton on Windows)

`ranx` won't install there, because `lz4` and `fastparquet` have no Windows builds for 3.15 (Nov 24), so score on the laptop.

**Order of work**
1. Run the bundled example.
2. Make the 10-K changes one at a time, running the repo's tests after each.

## Environment

- **Laptop:** Windows 11, no NVIDIA GPU. It runs every step that needs no GPU, in `.venv`.
  - **Smart App Control is on.** It blocks unsigned programs, and uv's own Python builds are unsigned: a uv-downloaded 3.15.2 failed to load. So `.venv` uses the signed python.org **Python 3.14.0** (built Nov 24), with CPU-only torch.
  - Always give uv the python.org interpreter's path, and never let it download a Python. Never suggest turning Smart App Control off: it can't be turned back on without reinstalling Windows.
  - Python 3.15 would also fail here, because `ranx`'s `lz4` and `fastparquet` have no Windows builds for 3.15 yet (Nov 24).
  - Importing `torch` and `transformers` takes about 35 seconds on this laptop.
- **Colab (GPU):** paid through the user's Google AI Pro plan, about 200 compute units a month. The project should need well under 100.
  - **L4** (24 GB, bf16, about 1.7 units an hour): about the school GPU's speed. Use it for question writing, embeddings, the reranker and inference.
  - **A100** (40 GB, about 5.4 units an hour; the 80 GB version about 7.5): about 4–5× faster. Use it for training when Colab offers it.
  - **T4** (16 GB, free): no bf16, and NV-Embed-v2 doesn't fit. Quick tests only.
  - These rates are third-party measurements from March 2026; check the balance in Colab.
  - Claude Code runs on the laptop. It can edit code and notebooks but can't run anything on Colab.
- **What fits on an L4 (24 GB):** QLoRA on Mistral-7B (batch size 1, gradient checkpointing); Qwen2.5-14B in 4-bit; NV-Embed-v2 for inference. Mixtral 8x7B doesn't fit.
- **Speed:** time a short run before planning long ones.
- **Fallback GPU:** a school computer with an NVIDIA RTX 4000 SFF Ada (20 GB VRAM, 70 W), driver 596.71, Windows, no admin rights, Python 3.15.2.

## Rules

- Never commit anything under `data/raw/`, `data/parsed/`, `checkpoints/` or `papers/`; they are gitignored. Do commit `data/questions/`, especially the hand-written questions, which can't be regenerated.
- **SEC EDGAR:**
  - Every request needs a User-Agent containing a name and email, and must stay under 10 requests per second.
  - Read the identity from the `SEC_IDENTITY` environment variable. Don't commit a real email.
- **Hugging Face:**
  - Read the token from the `HF_TOKEN` environment variable. Never put it in code, `config.yaml` or any committed file.
  - Mistral-7B-Instruct-v0.2 is not gated (checked Nov 23), so it downloads with the token alone.
  - The only API use is the bundled example's question writing (step 2.1). Use temperature 0.
- **Colab and Google Drive:**
  - Keep tokens out of notebook cells and files. Use Colab Secrets in the browser, or `getpass` in the VS Code extension.
  - Commit notebooks with their outputs cleared.
  - Drive holds parsed filings, generated questions, checkpoints and run outputs. Generated questions are also copied into `data/questions/` and committed.
- When `docs/OBJECTIVES_PLAN.md` or `docs/STAIR_Project_Overview.html` changes, regenerate both PDFs with `.venv\Scripts\python.exe docs\build_pdfs.py`, or say that the PDFs are out of date. When the scope changes, check that the overview still matches the plan.
- The user is a financial engineering student. Explain technical terms the first time they come up, and ask before committing or pushing.
