# How to complete each objective

This is the companion to `STAIR_Project_Overview.pdf`. For each objective it gives:
- what the objective answers
- the steps to complete it
- what you will have when it's done

Time estimates are in 8-hour working days and are summed up in the **Time** section at the end. `PLAN.md` has the same work as a checklist.

## Changes after the critical review (Nov 25)

A review of the plan found real gaps. With about 15 days left, these changes were made:

| Change | Why | Where |
|---|---|---|
| **STAIR trains on FY2024 filings only;** every test, the headline included, uses FY2025 | Training on the same FY2025 filings as the headline test invites the charge of leakage. Now no tested filing was trained on, the headline measures last year's model on this year's filing, and it takes one training run, not two | 1.10, objectives 2–4 |
| **Questions are generated per section** (3–6 for training, 2 for testing), about 5,000 for training | Per paragraph, long boilerplate sections would flood training, and ~30,000 questions would be too slow to generate and train on | 1.7 |
| **Question writing and judging run on vLLM** | Much faster than `transformers` in 4-bit | 1.7, Setup |
| **The hand-written set is about 200 questions:** 7 FinanceBench templates, the 8 checklist templates, 3 company-specific questions per company | Labelling ~450 questions would take 25–40 hours | 1.9 |
| **BM25 can use chunks too,** and all chunks overlap by 50 words; each section scores as its best chunk (MaxP) | So every RAG method works at the same chunk level as the reranker | 3.2–3.3 |
| **Training targets are always one section,** and the trie gets a unit test before training | So restricted decoding matches what the model learned, and tokenizer mismatches are caught early | 1.8, 2.7 |
| **Objective 5 caps each section at about 3,000 tokens,** keeping a long section's best chunks | Whole sections could swamp the answering model's prompt | 5.2 |
| **The untuned strong LLM with the ToC is named as the main threat** | A strong untuned model may come close to STAIR on such a standard structure | 3.4 |
| **Heading parsing is time-boxed** | Nested headings vary a lot between filers | 1.3 |
| **Pooled sections for the hand-written set are judged by the LLM judge,** with 100 decisions checked by hand | Judging them all by hand would take more than 10 hours | 3.6 |
| **The SearchTome reproduction and the FY2025-trained model are optional,** and objective 6 moves after the project | To fit 15 days | 2.3, 4.3, objective 6 |

One review point was not adopted: it said Python 3.14 lacks Windows builds for `lxml`, `pyarrow` and `ranx`. On Nov 24 they all installed from ready-made builds on 3.14, and they import cleanly.

---

## Before you start (about 1 day)

1. **Sector and companies (done, Nov 17).**
   - Technology: 8 training companies, 2 held-out first-time filers (CRWV, FIG), and 2 established test-only companies (ORCL, AMD). They're listed in `config.yaml`.
   - The test-only companies (added Nov 23) separate "a company the model never saw" from "a company's first 10-K". Both have years of filings; neither is trained on.
   - Fiscal years differ by company, so each company's own FY2024 and FY2025 are used. The held-out and test-only companies use FY2025 only.
   - **STAIR trains on the 8 training companies' FY2024 filings only, and every test uses FY2025** (Nov 25). Each FY2025 test question has a `group`: `seen_company` (the 8, in a new year), `first_time` (CRWV, FIG) or `established` (ORCL, AMD).
2. **Set up the laptop and Colab** by following "Setting up the laptop and Colab" below. The laptop part is done (Nov 24).
3. **Set up the project folder** so every later step has a place to write:

   ```
   data/raw/          downloaded 10-K HTML, one folder per company-year
   data/parsed/       toc.json + content.jsonl per filing
   data/questions/    train / dev / test files
   src/               parser, question generation, retrievers, training, scoring
   notebooks/         the Colab notebook that starts each GPU step
   checkpoints/       LoRA adapters
   results/           result tables (CSV) and figures
   ```

4. **Keep one config file** (`config.yaml`) with the company list, fiscal years, model names and random seed. Every number in the write-up should be reproducible from it.

## Setting up the laptop and Colab

**Where each step runs** (decided Nov 24; `runs_on` in `config.yaml`):

| Where | What runs there |
|---|---|
| Laptop (CPU, `.venv`) | downloading and parsing (1.1–1.6); the hand-written questions and their labels (1.9); BM25 (3.2); merging the hybrid's rankings (3.3a); calling the API baselines (3.4–3.5); judging pooled sections for the hand-written questions (3.6); scoring and statistics (3.1, 3.7–3.9); the WRDS checks (5.4); the write-up |
| Colab GPU | question writing and every LLM judge (1.7–1.8, 3.6, 5.6), with vLLM; STAIR training and inference (objective 2, 4.3); the untuned Mistral (3.4); NV-Embed-v2 and the reranker (3.3, 3.3b); objective 5's answering model (5.2–5.3) |
| Hugging Face API | the stronger LLM (3.4) and the whole-10-K method (3.5) |

**How work moves between the laptop and Colab.** Every step is a script in `src/` that reads files and writes files. No step depends on what's held in a notebook's memory, so no notebook ever switches between the laptop and Colab.
- **Code** goes through GitHub: push from the laptop, pull in Colab.
- **Data** goes through Google Drive. The Drive desktop app syncs `My Drive/stair/` on the laptop, and Colab sees the same folder at `/content/drive/MyDrive/stair/`.
- **GPU work comes in about three sessions:**
  1. question writing and judging
  2. training
  3. every GPU method on every test set

  Everything between them happens on the laptop.

**A. The laptop (done Nov 24).**
- **Machine:** Windows 11, with no NVIDIA GPU.
- **Smart App Control is on.** It's a Windows 11 security feature that blocks programs without a publisher's signature. uv's own Python builds aren't signed, so they don't run here.
- **Python:** `.venv` uses the signed python.org **Python 3.14.0**. Python 3.15 would also fail here, because `ranx`'s dependencies `lz4` and `fastparquet` have no Windows builds for 3.15 yet.
- **To rebuild `.venv`,** run these in this repo's folder in PowerShell:

  ```
  uv venv --python (py -3.14 -c "import sys; print(sys.executable)")
  uv pip install -r requirements.txt ipykernel
  ```

  A plain install gets CPU-only torch, which is right for a laptop without a GPU.
- **Set these user environment variables** (Windows Settings → "Edit environment variables for your account"):
  - `SEC_IDENTITY`: your name and email
  - `HF_TOKEN`
  - `PYTHONUTF8` = `1`: 10-K text is full of characters like "—" that break Windows' older default encoding
- **In VS Code,** choose the kernel **Python Environments… → .venv** for any notebook on the laptop.

**B. Google Drive.**
- Install Google Drive for desktop on the laptop.
- Make `My Drive/stair/` with these folders: `parsed/`, `questions/`, `checkpoints/`, `runs/` (each method's ranked sections) and `logs/`.
- **Never put WRDS data or tokens there.**
- **Models don't go there either.** Colab downloads them from Hugging Face in each session.

**C. Colab.**
- **Units:** Google AI Pro includes about 200 Colab compute units a month. Open a notebook at colab.research.google.com with the same Google account, and check the balance under Runtime → View resources.
- **GPU:** choose it under Runtime → Change runtime type.
  - **L4:** most GPU work.
  - **A100:** training, when Colab offers one.
  - **T4:** quick tests only. It has no bf16, and NV-Embed-v2 doesn't fit.
- **Cost:**
  - An L4 burns about 1.7 units an hour and an A100 about 5.4 (third-party measurements, March 2026). The project should need about 100 units.
  - Units burn while a runtime is connected, so end every session with Runtime → Disconnect and delete runtime.
- **Secrets:** add `HF_TOKEN` and `GITHUB_TOKEN` in the notebook's Secrets panel (the key icon). `GITHUB_TOKEN` is a read-only token for this repo and your private copy of the official code. Never type a token into a cell.
- **Long runs:** Google AI Pro has no background execution, so keep the tab open and the laptop awake. Checkpoints on Drive limit what a disconnect costs.
- **Record for every session:** the GPU type, Colab's Python version (`!python --version`) and the units used.

**D. The Colab notebook.**
- Keep one notebook, `notebooks/colab_gpu.ipynb`, run it in the browser, and commit it with its outputs cleared.
- Its first cells set up the session:

  ```python
  from google.colab import drive, userdata
  import os
  drive.mount("/content/drive")
  os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
  gh = userdata.get("GITHUB_TOKEN")
  !git clone https://{gh}@github.com/munashemacdonald/STAIR.git
  !git clone https://{gh}@github.com/<you>/<private copy of stair-clean>.git
  !pip install -r STAIR/requirements.txt
  ```

- Each later cell runs one script, reading from and writing to `/content/drive/MyDrive/stair/`.
- Colab already has a CUDA build of torch, and the install keeps it, so never reinstall torch there.

**E. Check the GPU in the notebook.**

```
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
!python -m bitsandbytes
```

- The second line should print `True` and the GPU you chose, for example `NVIDIA L4`.
- The third should report that bitsandbytes found CUDA.

**F. Set up your copy of the official code.**
1. **Make a private copy, not a GitHub fork.** A fork of a public repo is always public, and pushing your changes to it would publish them before the authors have given permission (see "Licences").
   - On GitHub, use "Import repository" with [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean)'s URL, and set it to Private.
   - Start your work from commit `3729c54`.
2. **Clone it on the laptop** next to this repo, so you can edit it in VS Code. Push your changes, and Colab pulls them.
3. **Add `"bitsandbytes"`** to its `dependencies` in `pyproject.toml`. Nothing else changes yet: on Colab, which runs Linux, `torchrun` and Colab's torch work as the authors wrote them.
4. **Install it in Colab** without reinstalling torch, and repeat check E.

**Done when:** the bundled example trains and evaluates on a Colab GPU (objective 2, step 2.1), and Colab's resources panel shows GPU memory in use during training.

## How the objectives depend on each other

- **1 (data)** comes first. Everything else uses it.
- **2 (train STAIR)** needs the FY2024 training questions from 1.
- **3 (compare)** needs 1 and 2.
- **4 (new filings)** is mostly answered by 3, because every test is on FY2025. Its breakdowns need 3's results. Its optional retrained model needs FY2025 training questions.
- **5 (investment use)** needs both years from 1, plus 2. It does not need 3 or 4.
- **6 (scale)** reruns 1–4 on more companies, after this project.

## Licences

Checked Nov 23. Say in the write-up what each part is licensed under.

| What | Licence | What it means here |
|---|---|---|
| 10-K filings (SEC EDGAR) | Public | Free to use, and to publish what you build from them |
| Official STAIR code | None given | Use it in your own private copy, and ask before publishing changes (requested Nov 19). A GitHub fork would be public, so don't use one. Keep your changes as a separate module or patch, so this repo holds only your own code |
| SearchTome | CC BY-SA 4.0 | Credit it. Anything you share that's built from it uses the same licence |
| FinanceBench templates | CC BY-NC 4.0 | Credit FinanceBench. Non-commercial only |
| Mistral-7B-Instruct-v0.2 | Apache 2.0 | Credit only |
| Qwen2.5-14B-Instruct and its AWQ version (writes the questions, judges) | Apache 2.0 | Its outputs can be used to train STAIR |
| NV-Embed-v2 | CC BY-NC 4.0 | Non-commercial only |
| bge-reranker-v2-m3 | Apache 2.0 | Credit only |
| API models (stronger LLM, whole 10-K) | Varies by model | Use their outputs only as baselines or for judging. Before any API model's output becomes STAIR training data, check that its licence allows training another model |
| WRDS data (objective 5) | Your subscription | Use it only to check results, on the laptop. Never commit or publish it, or upload it to Colab or Drive |

**Overall:** FinanceBench and NV-Embed-v2 make the project non-commercial. That's fine for a university project; say so in the write-up.

---

## Objective 1: Build a 10-K question dataset (about 5 days)

**What it answers:** nothing on its own. It builds the material every other objective uses. It is also where most of the risk sits: 10-K headings aren't marked up consistently, so the parser needs the most care.

### Steps

**1.1 Download the filings.**
- Use `edgartools` to download each company's FY2024 and FY2025 10-K from EDGAR. Skip 10-K/A amendments.
- Set your identity (name and email) first, because the SEC requires it, and stay under 10 requests per second.
- Save the main HTML document to `data/raw/<TICKER>_<FY>/`.
- For the held-out (CRWV, FIG) and test-only (ORCL, AMD) companies, download the FY2025 10-K only.
- Check whether the company puts its financial statements in Exhibit 13 instead of Item 8 (the filing will say "incorporated by reference"). If it does, download that exhibit too.

**1.2 Split each filing into Items.**
- The Items that matter most are 1, 1A, 1B, 1C, 2, 3, 5, 7, 7A, 8 and 9A.
- `edgartools` can return the text of each Item. Check its output against the filing's own table of contents.
- Skip the signatures and the exhibit index. Skip the cover page too, unless you keep it for FinanceBench question dg04 (see 1.9).

**1.3 Find the headings inside each Item.**
- 10-K HTML rarely uses real heading tags. Headings are usually short lines (under about 12 words) that stand alone as a paragraph and are entirely bold, underlined or in a larger font.
- In Item 8, the notes start with patterns like "Note 7 – Debt" or "7. Debt".
- Write rules for these with `BeautifulSoup`.
- Limit the outline to three levels: Item → heading → subheading.
- **Time-box this to about 1.5 days.** If headings inside Items still aren't reliable by then, fall back to Items plus numbered Notes (and the MD&A headings if they work), and say so in the write-up. Nested headings vary a lot between filers, and this step can otherwise eat the schedule.

**1.4 Number every section and write `toc.json`.**
- Number each node by its position. Item 7's third heading becomes "7.3" and its first subheading "7.3.1". Item 1A's second heading becomes "1A.2".
- The model's target is the number plus the title, e.g. "7.3 Liquidity and Capital Resources".
- This copies the paper's numbered titles and solves a real problem: 10-Ks repeat headings like "Overview" in several Items, and numbering makes every target unique.
- Each node has `id`, `title`, `level`, `parent` and `is_leaf`.
- Also write the official code's format, so its training code can read your filings directly: `{"title": ..., "table_of_contents": [{"id", "section_num", "title"}]}`.

**1.5 Write `content.jsonl`.**
- Write one line per leaf section, with `id`, `title` and `text`.
- Convert tables to plain-text rows (cells separated by " | ") instead of dropping them. A lot of 10-K information is in tables.

**1.6 Check each table of contents by hand.**
- Open the filing next to its `toc.json` and fix missed or false headings.
- Budget 20–30 minutes per filing, so roughly a day for all 20.
- Note how many fixes each filing needed. That is the parser's error rate, which goes in the write-up and matters for objective 6.

**1.7 Generate questions with an LLM, per section.**
- **Write questions per leaf section, not per paragraph** (changed Nov 25). Per paragraph, long boilerplate sections (risk factors, accounting policies) would flood training while short notes got few questions. It would also produce about 30,000 questions, too many to generate and train on in time.
  - **Training: the 8 training companies' FY2024 filings.** Each leaf section gets 3 questions, plus 1 for every 500 words beyond the first 500, up to 6. Write each question from one 100–300-word passage, with the passages spread across the section. That should give about 5,000 training questions (`questions` in `config.yaml`); if the count is far off, adjust the numbers.
  - **Dev and test: every FY2025 filing.** Each leaf section gets 2 questions.
- **The model:** an open instruction model writes the questions.
  - The paper used Mixtral 8x7B. It needs about 25 GB even in 4-bit, which is too much for an L4, and Colab doesn't always offer an A100.
  - Use Qwen2.5-14B-Instruct on a Colab L4 (see `config.yaml`), or a larger model through the Hugging Face API.
  - Only use an API model whose licence lets its outputs train another model (see "Licences").
- **Serve it with vLLM** in the question-writing session: load `Qwen/Qwen2.5-14B-Instruct-AWQ`, the same model in 4-bit. vLLM batches requests and is much faster than `transformers` in 4-bit.
  - vLLM installs its own torch, which is fine because this session does nothing else.
  - If vLLM won't install, use `transformers` in 4-bit with batched generation.
  - The judges in steps 1.8, 3.6 and 5.6 run the same way.
- Write the questions to `My Drive/stair/questions/`, then copy them into `data/questions/` on the laptop and commit them.
- The official code's question generator works with any OpenAI-compatible endpoint, so it can run through the Hugging Face API as it is.
- In the prompt, ask for questions an analyst would ask, without reusing the heading's words. Otherwise the task turns into keyword matching.
- Label each question with its section's id. **A training question always has exactly one target section,** so the model only ever learns one-item lists (see step 2.7).
- **Keep the hand-written test questions out of generation.**
  - Never use the FinanceBench or checklist templates, or any company-specific test question, as examples in the prompt.
  - After generating, embed every generated question and drop any whose cosine similarity to a hand-written test question is above about 0.85.
  - Otherwise STAIR trains on the test set's style, and the hand-written set stops being an independent test.

**1.8 Filter the questions.**
- Remove exact duplicates and near-duplicates (embedding cosine similarity above about 0.9).
- Have the LLM check each question against its paragraph ("Can this question be answered from this paragraph?") and drop the ones it says no to.
- **Find other sections that also answer each question.** A 10-K repeats facts across Items (risk factors, MD&A, the notes), so a generated question often has more than one correct section, but it's labelled only with the section it came from. A method that picks an equally good section would be marked wrong.
  - **Dev and test questions:** take the top sections from BM25 and from embeddings, set aside the question's own section, and keep the 3 highest-ranked others.
  - **The judge:** show an LLM judge the question and the best-matching ~300-word chunk of each of those 3 sections, and ask: "Does this text, on its own, answer the question?"
    - Use the question-writing model (Qwen2.5-14B), not one of the methods being compared.
    - Use temperature 0 and cache every response.
  - **What to record:** add every "yes" section to the question's correct sections, as answering on its own (format in `data/questions/README.md`).
  - **Training questions:** check a random 10% the same way. If more than about 15% have another correct section, check all training questions and drop the flagged ones, because an ambiguous target teaches STAIR an arbitrary choice. Training questions never get a second target: they keep one section or are dropped.
  - **Check the judge:** read 50 of its decisions yourself and record how often you agree.
  - **This check alone would favour BM25 and embeddings,** because only their candidates get checked. Step 3.6 evens it out: after every method has run, each method's top 3 sections are pooled and judged the same way.
- Read 100 random questions yourself and count how many are good. If fewer than about 80% are, fix the prompt and regenerate.

**1.9 Build the hand-written test set: the headline test.**
- **Where the questions come from:** three sources, all written without seeing your tables of contents.
  - **FinanceBench templates:** FinanceBench (Islam et al., 2023) has 25 generic questions, written with financial analysts, that it asks of every company (its "domain-relevant questions", dg01–dg25). They're saved as templates in `data/questions/financebench_templates.jsonl`.
    - **7 of them are used** (chosen Nov 25, before any table of contents was parsed; `handwritten` in `config.yaml`): dg09, dg11, dg20 and dg22, which objective 5 needs, plus dg14 (operating margin), dg19 (cash flows) and dg24 (working capital). Those last three keep financial-statement questions in the set and apply to every tech company.
    - The other 18 mostly repeat these (quick ratio, several gross-margin variants), need the cover page (dg04), or fit few tech companies (dg25, inventory turnover).
  - **Checklist templates:** 8 analyst questions in `data/questions/checklist_templates.jsonl` (ck01–ck08). They were fixed before any table of contents was parsed. They cover what FinanceBench barely touches: supply chain, debt maturities, internal controls, accounting changes, outlook, cybersecurity and buybacks. Objective 5 reuses them.
  - **Company-specific questions:** 3 per company, from what an analyst would want to know (news, earnings calls). **Write all 36 before you start building the parser.** Building it means looking at tables of contents, and each company's questions must be written before you see its table of contents. Save them before labelling.
- **What to ask:** all 15 templates (7 FinanceBench + 8 checklist) of every company's FY2025 filing, held-out and test-only companies included, plus the company-specific questions. That's 12 companies and up to 216 questions, about 200 after skips.
- **Why about 200, not 450** (changed Nov 25): at 3–5 minutes a question, labelling 450 would take 25–40 hours. About 200 keeps it to roughly 10–15 hours.
- **Each question records its `group`:** `seen_company`, `first_time` or `established` (see "Before you start").
- **Why the company-specific questions matter:** a template's twelve company versions tend to rise or fall together, so they count as one piece of evidence, not twelve.
  - The 15 templates give 15 independent units, and the 36 company-specific questions add 36, about 51 in all.
  - That's enough for differences of roughly 15 points to show up. Smaller differences may not reach significance; report them with their confidence intervals.
- **What to skip:** any question that doesn't apply, e.g. inventory turnover for a software company. Note why you skipped it.
- **Write the labelling rule first,** in `data/questions/README.md`, and stick to it. A section counts as correct in one of two ways:
  - **Answers on its own:** a reader could answer the question from this section alone. Several sections can each do this; any one of them is enough.
  - **Needed together:** the answer needs this section plus others, and none of them answers alone. For example, a ratio whose inputs sit in the balance sheet and a note.
  - **Doesn't count:** cross-references ("see Note 12"), general boilerplate (a risk factor that mentions litigation in general terms), and sections that only support an answer another section gives. For "What drove operating margin change?", MD&A explains the drivers, so the income statement doesn't count.
- **How to label:** record both kinds for each question, as `answers_alone` and `needed_together`. Most questions will only have the first.
- **Label before you look:** label each question before you see any method's output.
- **Second labeller, before you label the rest:**
  - Have a classmate label about 30 questions, picked at random across templates and companies, with the same rule and without seeing your labels.
  - Compare your labels with theirs, settle the disagreements, and tighten the rule where it was unclear. Then label the rest.
  - In the write-up, report the agreement: how often your `answers_alone` sets match exactly, and how often they overlap.
  - If nobody is available, re-label the same 30 yourself at least three days later, without looking at your first labels. That's weaker, but better than nothing.
- **dg04 (registered debt securities):** this is answered on the 10-K cover page. Keep the cover page as a section, or skip dg04.
- **Company-specific questions:** follow FinanceBench's rules for its annotator-written questions: realistic, varied, and needing some reasoning.
- **Why this set is the headline:** these are your most realistic test, written without seeing your tables of contents. The synthetic questions are easier and share the training questions' style, so STAIR's synthetic scores are inflated.
- **In the write-up:** describe this test set, in the methods section, as 7 of FinanceBench's 25 domain-relevant questions (Islam et al., 2023), adapted to each company's FY2025 10-K, plus the 8 checklist and 3 company-specific questions per company, with section labels you added. Say which 7 and why, how many you skipped and why, and report the second-labeller agreement. That's what lets readers recognise and trust the test set.
- **Licence:** CC BY-NC 4.0. Credit FinanceBench, and keep any question set you publish from them non-commercial.
- **Time:** writing the 36 company-specific questions and labelling about 200 questions takes roughly 1.5–2 days, plus a couple of hours for the second-labeller check.

**1.10 Split into train, dev and test** (changed Nov 25: the split is now by year).
- **Training, `train.jsonl`:** every FY2024 question of the 8 training companies. STAIR never trains on an FY2025 filing.
- **Dev and synthetic test:** the 8 training companies' FY2025 questions, split by section. 20% of sections go to `dev.jsonl` and 80% to `test_synthetic.jsonl`, so no section is in both.
  - Dev comes from FY2025 so that the checkpoint and every other choice are made in the same setting as the test: last year's model on this year's filing.
- **Held-out and test-only companies:** all their FY2025 questions go into `test_heldout.jsonl`. None are used for training.
- **Your hand-written questions:** these go into `test_handwritten.jsonl` and are never used for training.
- **Every FY2025 question** has a `group` field: `seen_company`, `first_time` or `established`.

### Result when complete

- **Parsed filings:** 20 checked filings in `data/parsed/` (8 companies × 2 years, plus FY2025 for the 2 held-out and 2 test-only companies), each with `toc.json` and `content.jsonl`.
- **Question files:**
  - `train`: FY2024, about 5,000 questions
  - `dev` and `test_synthetic`: FY2025, the 8 training companies, split by section
  - `test_handwritten`: about 200 FY2025 questions. These are 7 FinanceBench templates, the 8 checklist templates, and 3 company-specific questions per company. Labels come with a second-labeller agreement figure.
  - `test_heldout`: FY2025, the 4 held-out and test-only companies
- **Parser:** `src/parse_10k.py`, which turns any 10-K into those two files with one command.
- **Ambiguity report,** for each test set:
  - how many questions have more than one correct section
  - how many are "needed together"
  - how often you agreed with the judge on the 50 decisions you checked
- **Dataset table** for the write-up:

| Company | Year | Leaf sections | ToC length (tokens) | Train | Dev | Test | Parser fixes needed |
|---|---|---|---|---|---|---|---|

---

## Objective 2: Train STAIR on the filings (about 2 days, much of it GPU time)

**What it answers:** does STAIR, run with the authors' own code, work on 10-K filings? It trains on the 8 training companies' FY2024 filings only; objective 3 tests it on FY2025.

**Build on the official code.** The authors' code is public: [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean), commit `3729c54`. Build on it instead of writing training code from scratch.
- In the write-up, call it the "official implementation", not a "baseline".
- The repo has no licence file. Keep it in your own private copy until the authors give permission, and list every change you make.

All of this objective runs on Colab. You edit the code on the laptop, push it, and pull it in the Colab notebook.

### Steps

**2.1 Run the official code unchanged, on Colab.**
- Set up your copy as in "Setting up the laptop and Colab", step F. Colab runs Linux, so the code runs as the authors wrote it; only `bitsandbytes` is added. Change nothing else yet.
- The example's `prepare-data` step writes questions through an OpenAI-compatible LLM endpoint. Point it at the Hugging Face API with these variables, set from Colab Secrets:
  - `STAIR_LLM_API_BASE=https://router.huggingface.co/v1`
  - `STAIR_LLM_API_KEY`: your Hugging Face token
  - `STAIR_LLM_MODEL`: a model the API serves
- Run the bundled example (`prepare-data` → `train` → `eval`) on an L4. Check Colab's resources panel during training to confirm the GPU is in use.

**2.2 Add 4-bit loading.**
- The code loads models in bf16 only. Mistral-7B-Instruct-v0.2 in bf16 takes about 14.5 GB before any training overhead, which is too tight for an L4's 24 GB with long table-of-contents prompts.
- Load it in 4-bit (`bitsandbytes`, NF4) and keep the paper's LoRA settings (rank 16, alpha 32).
- Use 4-bit on every GPU, A100 included. Colab doesn't always offer the same GPU, and every run should use the same setup.

**2.3 Reproduce one SearchTome domain (optional since Nov 25, if time allows).** The bundled example in step 2.1 already shows the code runs. This step checks it reaches the paper's accuracy.
- Train on the finance domain from Hugging Face (`meghanadh/SearchTome`) with Mistral-7B and the settings you'll use for 10-Ks.
- Compare your R@1 with the paper's 82.8 for finance. The public release has one book per domain while the paper used three, so expect some gap.
- If the 10-K results later look poor, this tells you whether the cause is your changes or the base code.

**2.4 Adapt the code for 10-Ks.**
- **Many filings per run:** the code reads one table of contents per dataset. Change the data loading so each question carries its filing's id and is paired with that filing's ToC, in training and evaluation.
- **Prompt:** replace "Book: {title}" with the company and filing, e.g. "NVIDIA Corp., Form 10-K, FY2025". Keep the rest of the prompt.
- **Input:** skip the PDF step. Your parser (objective 1) already writes the ToC in the code's format.
- **Length:** measure the longest ToC plus question in tokens, and raise `max_seq_length` (2,048 by default) above it.
- Run the repo's tests after each change.

**2.5 Check the training time against a budget, and save often.**

**Why a budget:** every training example carries the filing's whole table of contents, likely 2,000–6,000 tokens for a 10-K. Thousands of examples over several epochs on an L4, which is about as fast as the school's 70 W card, can take days, and the optional retrained model (step 4.3) would need a second run. The paper allowed up to 200 epochs with early stopping; that won't fit.

**Estimate before the full run:**
- **Tokens per epoch** = number of training questions × average prompt length in tokens (ToC + question + answer). Your dataset table from objective 1 already has the ToC lengths.
- **Speed:** train for about 200 steps and read the tokens per second from the log (tokens per step ÷ seconds per step).
- **Hours per epoch** = tokens per epoch ÷ tokens per second ÷ 3,600.
- **Dev evaluations count too.** With the default metric (dev loss) they're quick. Dev Recall@1 needs generation, so run it on the 300-question sample and only at checkpoints.

**Budget:** one full training run should take at most about 8 hours (`training.max_hours_per_run` in `config.yaml`). Within that, cap the epochs (likely 3–10) and keep early stopping.
- **Measure the speed on the GPU you'll train on.** An A100 is about 4–5 times faster than an L4, so the same run might take about 2 hours there.
- **Compute units:** hours × the GPU's rate. An 8-hour L4 run uses about 14 units, and a 2-hour A100 run about 11. Check that the run fits in the units left, with room for the optional retrained model.

**If the estimate is over budget, cut in this order,** re-estimating after each step. Questions are already capped per section (step 1.7), so that's no longer a cut.
1. **Use a two-level table of contents** (Item → heading), which shortens every prompt. Each question then moves to its section's level-2 parent, hand-written labels included. Relabel before training, and say so in the write-up.
2. **Train fewer epochs,** for example 3.
3. **As a last resort, use a smaller model** (1–4B parameters). Report it as a deviation from the paper.

Record the estimate and the actual time for each run.

**Save often, and make training resumable.** Colab can stop a run partway through: the session time limit, an idle timeout, a closed tab or a sleeping laptop. Google AI Pro has no background execution, so keep the tab open and the laptop awake during a run.
- **Save a checkpoint about every 30 minutes, to Google Drive.**
  - Point the run folder (`--out`) at `/content/drive/MyDrive/stair/checkpoints/<run>`, never at `/content`, which Colab wipes when the session ends.
  - The code saves every N steps (`save_interval`, 30 by default). From the 200-step timing, set N to the number of steps that take about 30 minutes (`training.checkpoint_every_minutes`).
  - The best checkpoint is chosen at the end (`load_best_model_at_end`), so the Hugging Face trainer needs saving to fall on evaluation steps. Set `eval_interval` to the same N, or to a number that divides it.
  - Keep `save_total_limit` at 3. The trainer also keeps the best checkpoint. Each checkpoint (adapter plus optimizer state) is about half a gigabyte.
- **Make training resumable.**
  - The official `train` command always starts from scratch. Add a `--resume` option that passes `resume_from_checkpoint=True` to the trainer.
  - The trainer then continues from the latest checkpoint in the run folder, with the same step count, learning-rate schedule and optimizer state.
  - Test it once on the bundled example: stop a run after its first checkpoint, then resume it.
- **After an interruption,** resume rather than restart, and note in the training log when the run resumed.

**2.6 Pick the best checkpoint.**
- The code keeps the checkpoint with the lowest dev loss. The paper chose by dev Recall@1.
- Switch to dev Recall@1 if time allows. Either way, say in the write-up which one you used.

**2.7 Measure invalid outputs, then add the restriction as your extension.**
- **Official evaluation:** the code generates freely (greedy) and counts titles that aren't in the ToC afterwards. Keep that as the faithful replication, and report its invalid-output rate (the paper reported 0.05%).
- **Your extension: restrict the output in the format the model was trained on.**
  - The model learns to answer with a list, such as `["7.3 Liquidity and Capital Resources"]`, not a bare title. A trie (a prefix tree of the allowed token sequences) built from bare titles would block the opening `["`, forcing a format the model never saw.
  - So, for each filing, build the trie from the full strings `["<section_num> <title>"]`, one per leaf section, each followed by end-of-sequence.
  - **Tokenize them exactly like the training targets**, at the same position after the prompt. Don't tokenize the titles separately and join the pieces: the tokenizer splits quotes and spaces differently in context (for example `▁Item` with a leading space versus `Item`).
  - Pass the trie to `generate` through `prefix_allowed_tokens_fn`.
  - Every training target is a one-item list (step 1.7), so a trie over one-item lists matches what the model learned. A trie over lists of several sections isn't needed.
- **Test the trie before training.** Write a unit test with the base model's tokenizer, which needs no GPU:
  - for 20 training examples, the target's token ids, tokenized as in training, must be exactly one path through that filing's trie, ending in end-of-sequence
  - the trie must accept no title that isn't in the filing

  This catches tokenizer mismatches before they cost a training run.
- **Top 3 for each version:**
  - **Official:** the first 3 items of the generated list. Invalid titles are dropped, as the official code does, and counted separately. Every training target names one section, so the model will usually return a one-item list; then its R@3 equals its R@1. Report that rather than padding the list.
  - **Restricted:** beam search with 5 beams, keeping the 3 best. Each beam is a one-item list, so the top 3 are three different real sections, ranked by beam score.
- Report both versions. With the restriction on, invalid outputs are zero by design.

### Result when complete

- **Model:** the best LoRA adapter, trained on FY2024, in `checkpoints/stair_fy2024/` (on Drive).
- **Trie unit test:** passing, before training.
- **Reproduction check (if done):** your SearchTome finance R@1 next to the paper's.
- **Code:** your private copy of the official code, with a list of every change, for the write-up.
- **Figure:** the dev metric over training.
- **Retrieval script:** give it a company and a question, and it returns the top 3 sections.
- **Recorded numbers:**
  - dev R@1
  - invalid-output rate, unrestricted and restricted
  - estimated and actual training time for each run, plus any budget cuts made (step 2.5)
  - the GPU type, GPU memory used and compute units used for each run

---

## Objective 3: Compare STAIR with the alternatives (about 2.5 days)

**What it answers:** the main research question. Does STAIR find the right 10-K section more often than standard RAG retrieval, and at what cost?

None of the comparison methods is trained; only STAIR is fine-tuned (decided Nov 9). The paper's STAIR vs. DSI result is cited, not reproduced: 82.6 vs. 76.9 average R@1 on SearchTome, with invalid outputs of 0.05% vs. 3.25%.

**Every method is tested on the FY2025 filings.** STAIR was trained on FY2024 only. The RAG methods index the FY2025 filings directly, as a real system would.

### Steps

**3.1 Write the scoring script first.**
- Every method produces the same output: for each question, a ranked list of section ids.
- **Where STAIR's ranked list comes from:** see step 2.7. The official version uses the first 3 valid items of its generated list; the restricted version uses its 3 best beams. Every other method returns a ranked list directly.
- **What R@k means here:** a question counts as found at k if any of its correct sections is in the top k. This is "hit@k", the official code's `match@k`.
  - "Correct" means `answers_alone` plus `pooled_answers_alone`, the sections added after the methods run (step 3.6).
  - Also score without the pooled sections, and report both.
  - With one correct section, it equals ordinary recall, which is how the paper reports R@1. So your numbers stay comparable with the paper's.
  - Don't use the official code's `recall@k` as the headline. It divides by the number of correct sections, so a question with 3 correct sections can score at most 0.33 at k = 1, even when the top answer is right.
- **Needed-together questions:** report these separately, because no single section answers them, so leave them out of R@1. Use two measures:
  - **All@3:** whether every needed section is in the top 3.
  - **The share of needed sections found in the top 3:** this is the official code's `recall@3`.
- **nDCG@3:** count every labelled section, of both kinds, as relevant.
- Use `ranx` where it fits.
- Test the script on a made-up run whose scores you already know, including questions with several correct sections and a needed-together question.
- Score every method with this one script, STAIR included. The official code's eval reports the same metrics, so check the two agree.

**3.2 BM25 (keyword RAG).**
- Use `rank_bm25`, with a separate index per filing, and search only the filing of the company the question is about.
- **Try two versions on dev, and keep the better one for test** (added Nov 25):
  - whole leaf sections
  - the chunks from step 3.3, with each section scored by its best chunk

  The hybrid uses whichever wins.

**3.3 Embedding retrieval (dense RAG).**
- **Cut every section into chunks** of about 300 words, overlapping by 50 words, each starting with its section number and title. BM25 (3.2) and the reranker (3.3b) use the same chunks.
- **Embed the chunks.** NV-Embed-v2 matches the paper; `bge-large-en-v1.5` is a lighter option.
- **Score each section by its best-matching chunk.** This is called MaxP. A whole long section squeezed into one embedding would blur its meaning, and MaxP avoids that.

**3.3a Hybrid: BM25 + embeddings, merged with reciprocal rank fusion.**
- **What:** combine the rankings from steps 3.2 and 3.3 into one. These are the paper's own two RAG baselines, combined the way current RAG systems usually do it.
- **How:**
  - Reciprocal rank fusion (RRF) gives each section 1 / (60 + its rank) from each method, adds the two, and sorts by the total. 60 is the standard constant (Cormack et al., 2009).
  - Fuse each method's full ranking of the filing's sections.
  - `ranx` does this with `fuse(..., method="rrf")`.
- **Why it helps:** the two methods make different mistakes. BM25 finds exact phrases ("material weakness", "Item 1C"). Embeddings find rephrasings ("quick ratio" → the balance sheet). Merged, they usually beat both.
- **Cost:** almost none, since it reuses rankings you already have.

**3.3b Hybrid + reranker.**
- **What:** run the cross-encoder `BAAI/bge-reranker-v2-m3` over the hybrid's top 20 sections, and re-sort them by its score.
  - A cross-encoder reads the question and the text together, which is more accurate than comparing two separate embeddings.
  - Retrieving first and then reranking is how current RAG systems are usually built, and this is the strongest version that needs no training.
- **How:**
  - Use the same chunks as step 3.3 for each of the 20 sections.
  - Score every question–chunk pair, with a maximum length of 512 tokens. A section's score is its best chunk's score.
  - Load it with `sentence-transformers` (`CrossEncoder`), which is already in `requirements.txt`.
  - It has about 570 million parameters, roughly 1 GB on the GPU. Run it after NV-Embed-v2 has been unloaded, not alongside it.
  - Its licence is Apache 2.0.
- **Report the ceiling:** the reranker can only reorder what the hybrid found. Report the hybrid's R@20, how often a correct section is in its top 20: that's the most the reranker can reach.
- Record its seconds per question with the other methods (step 3.8).

**What these two don't fix:** STAIR is still the only method trained on questions about these filings. That's the Nov 9 decision, and the paper's setup too.
- Say so in the write-up: the comparison is STAIR against the best untrained RAG pipeline.
- A reranker fine-tuned on STAIR's training questions would be the natural trained competitor. List it as future work.

**3.4 LLM given the ToC, without fine-tuning.**
- Send the STAIR prompt to the untuned Mistral, on the same GPU type and 4-bit setup as STAIR. The paper got 13.8% R@1 this way.
- Send it to one stronger, 70B-class model through the Hugging Face API as well. The paper used Claude Haiku 4.5.
- Count invalid outputs for both.
- **Record the stronger model's training-data cutoff,** from its model card. If a filing was published before that date, the model may have read that 10-K during training, which would give it an advantage you can't see (step 3.9 checks this). Mistral-7B-v0.2's data predates every filing here.
- **This is the comparison most likely to challenge fine-tuning** (noted Nov 25).
  - Every 10-K follows the SEC's fixed Item structure, and its table of contents fits in a few thousand tokens. So a strong untuned model may come close to STAIR, especially on companies STAIR never saw.
  - If it does, STAIR's case rests on what the strong model can't offer: it runs locally (private filings, no per-question cost, steady speed), and in its restricted version it never names a section that doesn't exist.
  - Say so in the write-up rather than resting the case on R@1 alone.

**3.5 LLM given the whole 10-K.**
- Send the full filing text, with each section marked by its id and title, and ask which section answers the question.
- A 10-K is often 50,000–100,000 tokens. This needs a long-context model through the Hugging Face API, and it's the slowest and most expensive method. Estimate the token cost before running it.
- Run it only on the hand-written questions plus a random sample of about 300 synthetic ones.
- **Score every other method on the same 300 synthetic questions too,** and report them as a separate table. Then this method is compared like for like. Its hand-written score needs no subset, because it runs on all of them.
- **Keep the cost down:** many providers discount a repeated start of a prompt ("prompt caching"). Put the filing text first and the question last, so every question about the same filing repeats the same start.
- **Record this model's training-data cutoff** too, as in step 3.4.

**3.6 Pool every method's answers, then rescore.**
- **Why:** step 1.8 only checked other sections that BM25 and embeddings suggested. Those two methods could get credit for an alternative correct section, while the alternatives other methods find, STAIR's included, were never checked. Pooling is the standard fix in retrieval benchmarks (it's how TREC builds its labels).
- **When:** after every method has run on the test sets, and before any significance test.
- **What to pool:** each method's top 3 sections for each test question, set aside any already labelled or judged, and keep one copy of each remaining section.
- **Who judges, without knowing which method found each section:**
  - **Generated test questions** (`test_synthetic`, `test_heldout`): the LLM judge from step 1.8, with the same model, prompt and temperature 0. Show the sections in random order, without method names.
  - **Hand-written questions:** the same LLM judge, given the labelling rule in `data/questions/README.md`.
    - Then check a random 100 of its decisions yourself, blind, with a small script that shuffles them and hides which method found each section. Report how often you agree.
    - Changed Nov 25: judging the whole pool by hand would take more than 10 hours.
- **What to record:** put every "yes" in `pooled_answers_alone`, not in `answers_alone`, so the labels made before any method ran stay separate.
- **What to report:**
  - every method's scores with and without the pooled sections
  - how many pooled sections each method contributed

  If one method gains much more than the others, say so.
- **Don't pool the dev set.** It's for choosing checkpoints, not for results.

**3.7 Significance test.**
- **Choose the primary comparison before running any test:** STAIR (official version) against hybrid + reranker, R@1 on the hand-written set, judged by the confidence interval of the difference (below). Record it in `config.yaml` (`scoring.primary`). The headline claim rests on this one comparison.
- For STAIR against each other method, run a paired randomization test on per-question R@1 (Smucker et al., 2007).
- `ranx` has it built in as Fisher's randomization test.
- Report the p-values.
- **Confidence intervals for the hand-written set:** don't resample single questions. Resample whole units: each of the 15 templates is one unit (all its company versions together), and each company-specific question is its own unit.
  - The twelve companies' answers to one template tend to rise or fall together. So about 180 template questions carry roughly as much independent evidence as 15. The 36 company-specific questions add 36 more units.
  - Treating them as about 200 separate questions would make the results look more certain than they are.
- **Confidence interval of the difference:** for STAIR against each other method, bootstrap the difference in R@1 (STAIR minus the other method), resampling the same units for both. This is the number the headline claim rests on. Differences vary less than raw accuracy, so they're more likely to be clear-cut.
- **Correct for many comparisons (Holm):** every comparison other than the primary one is secondary: STAIR against each method, on each test set.
  - Testing many comparisons at once makes some look significant by chance. The Holm method corrects for that. Sort the m p-values from smallest to largest. The smallest must be below 0.05 / m, the next below 0.05 / (m − 1), and so on. Stop at the first one that fails; it and every larger one are not significant.
  - Report both the raw and the adjusted p-values.
- **The other test sets have clusters too:**
  - **Synthetic questions** written from the same section rise or fall together, so resample sections, not questions.
  - **The held-out and test-only groups** have only 2 filings each. Report per-company results, with confidence intervals from resampling sections, and make no significance claims from them.
  - **The seen-company group** (the 8 training companies in a new year) is the largest group, and the one that carries the headline's statistics.
- **Small counts:** in any breakdown with fewer than about 30 questions (objective 4's new sections, the needed-together questions), show the count and the confidence interval. Don't draw a conclusion from that cell alone.
- **Freeze the test sets.** Make every choice (prompts, epochs, the checkpoint, thresholds) on dev, never on test. Record the date of the first test run. If anything changes after that, report the rerun and why.

**3.8 Cost and speed.** For each method, record:
- seconds per question, with the GPU type. Times only compare on the same GPU, so run every GPU method on the same type.
- cost per 1,000 questions, in Colab compute units (about $0.10 each) or API charges. BM25 and the hybrid's merge run on the laptop's CPU and cost nothing.

**3.9 Break down the results**
- **by test set and group:** synthetic and hand-written; and, within each, seen companies in a new year, held-out first-time filers (CRWV, FIG) and established test-only companies (ORCL, AMD).
  - The gap between the last two separates "a company's first 10-K" from "a company the model never saw".
  - The gap between seen and unseen companies shows how much training on a company's earlier filing helps.
- by Item: 1, 1A, 7 and 8
- by how much each section changed since FY2024 (objective 4)
- **by filing date, for the API models:** filings published before versus after each model's training cutoff (steps 3.4–3.5). A large gap suggests the model remembered those filings. Mention it in the write-up as a limitation either way.
- **by the number of correct sections:** one, or several
- **by template group,** for the hand-written set:
  - **Financial-statement templates:** dg14 (operating margin), dg19 (cash flows), dg22 (debt) and dg24 (working capital).
  - **All the others,** including the checklist and company-specific questions.
  - These templates point at the same three statements, which have similar titles in every filing. That favours methods that read section titles, and BM25 struggles there, because "working capital" rarely appears in a balance sheet's text.
  - **The headline claim should hold in both groups.** If STAIR wins only on the financial-statement questions, say so: that's a win on title-matching, not on understanding the filing.

This shows where STAIR helps and where it doesn't.

### Result when complete

The main results table of the project:

| Method | R@1 hand-written (headline) | R@1 synthetic | R@1 by group (seen / first-time / established) | R@3 | nDCG@3 | Invalid outputs | Sec / question | Cost / 1k questions | p vs. STAIR (Holm-adjusted) |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | | | | | | – | | | |
| Embeddings (dense) | | | | | | – | | | |
| Hybrid (BM25 + embeddings, RRF) | | | | | | – | | | |
| Hybrid + reranker | | | | | | – | | | |
| Mistral + ToC, untuned | | | | | | | | | |
| Stronger LLM + ToC | | | | | | | | | |
| LLM + whole 10-K | | | | | | | | | |
| **STAIR** (trained on FY2024) | | | | | | | | | – |

R@1 and R@3 are hit@k, as defined in 3.1. Needed-together questions get a small table of their own, with All@3 and the share of needed sections found.

**Like-for-like table:** every method's R@1 on the 300 synthetic questions the whole-10-K method also answered (step 3.5).

**The headline is the hand-written column**, with the confidence interval of the difference from step 3.7. Those questions were written without seeing the tables of contents. Synthetic results come second, because STAIR trained on questions in the same style. All scores are after pooling (step 3.6), and the scores before pooling go in an appendix table.

**The row to beat is hybrid + reranker.** Like STAIR, it runs on the same Colab GPU with no API, and it's the strongest RAG method in the table.

It also gives a one-sentence answer to the main question, for example: "On the hand-written analyst questions, STAIR's R@1 is X points above the best RAG method (95% CI …), and the gap holds for both template groups." If STAIR doesn't win, that is still a valid result, and the breakdown in 3.9 explains why.

---

## Objective 4: Test whether it holds up on new filings (about half a day)

**What it answers:** each company files a new 10-K every year. Does a model trained on last year's filings still work on this year's without retraining?

**Since Nov 25 the main experiment already tests this.** STAIR trains on FY2024, and every test in objective 3 is on FY2025. This objective breaks those results down.

### Steps

**4.1 Measure how much the ToCs changed.**
- For each of the 8 training companies, match each FY2025 title against its FY2024 ToC: exact match first, then fuzzy match for small renames (`rapidfuzz`).
- Label each FY2025 section as unchanged, renamed or new.

**4.2 Break down accuracy by that label,** for the seen companies' FY2025 questions, synthetic and hand-written. Expect errors to be concentrated in new and renamed sections.
- In STAIR's official version, count invalid outputs that are FY2024 titles separately. They show the model falling back on last year's filing.

**4.3 Optional, if time allows: the retrained model.**
- Train STAIR on the 8 companies' FY2025 filings as well. Generate FY2025 training questions as in step 1.7, and drop any within 0.9 cosine similarity of an FY2025 dev or test question.
- Test it on the same FY2025 test questions. The gap to the FY2024-trained model is the cost of not retraining.
- This is also the paper's own setup, training on the documents it searches. Report it as secondary, and say that its test filings were in its training data.

**4.4 Bring in the held-out and test-only companies.** Objective 3 already measured R@1 on companies STAIR never trained on, the other kind of "new filing". Report both groups here: first-time filers and established companies.

**Show the counts.** Put the number of questions in each column of the table below. The "new" column will be small; follow the small-count rule in step 3.7.

### Result when complete

| Model | Tested on | R@1 all | R@1 unchanged sections | R@1 renamed | R@1 new |
|---|---|---|---|---|---|
| STAIR trained on FY2024 | FY2025, seen companies | | | | |
| BM25 (for reference) | FY2025, seen companies | | | | |
| STAIR trained on FY2024 | held-out first-time filers (CRWV, FIG) | | | | |
| STAIR trained on FY2024 | established test-only companies (ORCL, AMD) | | | | |
| STAIR trained on FY2025 (optional, 4.3) | FY2025, seen companies | | | | |

It also gives a practical conclusion, for example: "Trained on last year's filings, STAIR's errors concentrate in new sections, so it should be retrained once each filing season."

---

## Objective 5: Show the investment use (about 2 days)

**What it answers:** what an analyst actually gets from STAIR. That is consistent, cited answers to the same questions across many companies and years.

It also asks whether STAIR's sections lead to better answers than the best RAG method's, checked against independent data where possible.

### Steps

**5.1 Write the checklist, with a scoring scale for each question.**

The checklist is already fixed:
- **The 8 checklist templates** in `data/questions/checklist_templates.jsonl` (ck01–ck08): customer dependence, supply-chain risks, debt maturities, material weaknesses, accounting changes, outlook, cybersecurity and buybacks.
- **Four FinanceBench templates:** dg09 (revenue concentration), dg11 (litigation), dg20 (customers) and dg22 (debt).

All 12 are also in the hand-written test set (step 1.9), so you already know how well each method retrieves the sections the checklist needs.

For each question, write a 1–5 scale in plain words. For customer dependence, for example: 1 = no customer above 10% of revenue … 5 = one customer above 30%. The written scale is what makes scores comparable across companies and repeatable.

**5.2 Run the checklist for every company and year, with two retrievers.**
- **STAIR gives its top 3 sections,** not just one. Checklist questions often need more than one part of a filing: customer dependence can sit in Item 1, Item 1A and a note.
  - STAIR was trained on the FY2024 filings, so for FY2024 it's searching filings it has seen. That's fine for building signals, but its FY2024 retrieval isn't a test result.
- **Cap each section at about 3,000 tokens** (`objective5.section_token_cap`; added Nov 25).
  - If a section is longer, keep its best chunks for the question, picked by the reranker from step 3.3b, up to the cap, in their original order.
  - Use the same rule for both pipelines, and record each prompt's length.
  - Whole long sections would swamp the answering model's prompt, and details buried in the middle of a long prompt tend to be missed.
- An answering LLM reads the 3 sections' text and returns JSON with:
  - a 2–3 sentence answer
  - the score on the scale
  - a supporting quote
  - the id of the section the quote comes from
- Use temperature 0. The answering model (`objective5.answer_model` in `config.yaml`) isn't one of the compared methods.
- **Then run the checklist again with hybrid + reranker's top 3 sections** instead of STAIR's, using the same answering model and prompt.
  - This is what shows whether STAIR adds anything. Without it, the signal table would look the same with any retriever.

**5.3 Add the year-over-year change.**
- Score change = FY2025 score − FY2024 score.
- For each question, also have the LLM read the FY2024 and FY2025 sections side by side and summarise what's new.

**5.4 Check the quality.**
- **Repeatable:** re-run each question reworded. Report how often the score stays the same or moves by at most 1.
- **Sensible:** before looking at the LLM's scores, score about 20 (company, question) pairs yourself on the same scale. Report exact agreement and agreement within 1 point.
- **Right section:** for the same 20 pairs, check that STAIR picked the right section.
- **Spread:** show how each question's scores are distributed. If every company gets the same score, that question can't tell companies apart.
- **Against independent data, where your WRDS subscription has it.** Your own scores aren't independent of the scale, because you wrote both. These sources record the same facts separately:
  - **Material weaknesses:** Audit Analytics' SOX 404 internal-controls data says whether each company reported one.
  - **Customer dependence:** Compustat's customer segment data lists major customers and their share of revenue.
  - **Debt maturities:** Compustat's annual data gives the debt due in each of the next five years (items DD1–DD5).
  - **Buybacks:** Compustat's purchases of common and preferred stock (PRSTKC).
  - For each, report how often each pipeline's score agrees with the WRDS figure.
  - If your subscription doesn't include a source, use your own scores for that question and say so.
  - Use WRDS data only for this check, on the laptop. Never commit or publish it, or upload it to Colab or Drive.
- **STAIR against hybrid + reranker:** for each question, compare the two pipelines on agreement with your scores and with WRDS. This result says whether STAIR's sections lead to better answers.

**5.5 Write 2 short case studies.** Take the biggest score changes, show the FY2024 and FY2025 text side by side, and explain what changed.

**5.6 Check the cited answers.** The project's goal promises cited answers, so measure them.
- **Pick about 50 hand-written questions** at random, spread across templates and companies.
- **Answer each one twice:** give the answering LLM STAIR's top 3 sections, then hybrid + reranker's, and ask each time for a short answer that cites the section it used.
- **Judge two things** for each answer:
  - **Correct:** does it answer the question? Judge against the question's labelled correct sections.
  - **Supported:** is every claim backed by the section it cites? An answer can be correct but cite the wrong section, or cite the right section and misread it.
- **Who judges:** an LLM judge (the question-writing model, temperature 0) judges all of them. Check 20 of its decisions yourself.
- **Report, for both pipelines:**
  - the share correct
  - the share supported
  - the share citing a section that exists
  - how often you agreed with the judge

### Result when complete

- **Signal table:** `results/signal_table.csv`, with one row per company × question. Each row has:
  - FY2024 score
  - FY2025 score
  - the change
  - the section each score came from
  - the supporting quotes
- **Heat map:** companies down the side, questions across the top, colour = FY2025 score, arrows for changes.
- **Quality numbers,** for example: "Scores agreed with mine on 15 of 20 pairs exactly, and 19 of 20 within one point."
- **Case studies:** 2 of them.
- **Pipeline comparison:** the checklist run with STAIR's sections and with hybrid + reranker's, side by side. For each question, it shows agreement with your scores and with WRDS.
- **Cited-answer check:** the correct, supported and valid-citation rates for both pipelines (step 5.6).
- **Scope sentence:** a clear statement that this shows how the signals are built, not that they predict returns.

---

## Objective 6 (after this project): Scale to 25–30 companies (3 days)

**What it answers:** when the number of companies triples, does the approach still work, and does the parser still work without hand-fixing?

It doesn't fit in the 15 days left on Nov 25. Keep it as next steps in the write-up.

### Steps

**6.1** Add 15–20 companies from the same sector. Check on EDGAR that each has the filings you need.

**6.2** Run the parser with no hand-fixing. Check a random 20% of the new ToCs by hand and record the error rate.

**6.3** Generate questions for the new companies, and retrain STAIR on the larger set.

**6.4** Rerun the comparison (objective 3) and the new-filings test (objective 4).

### Result when complete

- **Results table:** a new group of rows for 25–30 companies, next to the 10-company results.
- **Parser:** its error rate without hand-fixing.
- **Training:** how much training time grew.
- **Conclusion:** one sentence on whether accuracy held. STAIR only ever sees one company's ToC at a time, so it should hold up.

---

## Time

Re-planned on Nov 25, with about 15 days left.

| Part | Working days (8 h) |
|---|---|
| Setup still to do (Colab, Drive, the official code, environment variables) | 0.5 |
| Objective 1 | 5 |
| Objective 2 | 2 |
| Objective 3 | 2.5 |
| Objective 4 (breakdowns) | 0.5 |
| Objective 5 | 2 |
| Write-up | 2 |
| **Core total** | **14.5** |
| Optional: the retrained model (4.3) | 1 |
| Optional: the SearchTome reproduction (2.3) | 0.5 |
| Objective 6 | after this project |

- **Objective 1** includes:
  - parsing, time-boxed to about 1.5 days (step 1.3), and checking 20 tables of contents by hand (about a day)
  - writing 36 company-specific questions and labelling about 200 hand-written questions (1.5–2 days), plus the second-labeller check
  - generating and checking questions on Colab, mostly unattended GPU time (about half a day)
- **Objective 2** includes setting up the official code, the 10-K changes and the trie unit test. Training itself runs unattended; work on objective 3's laptop steps meanwhile.
- **Objective 3** includes the two hybrid methods (steps 3.3a–3.3b), and pooling every method's answers and judging the pooled sections (step 3.6).
- **Objective 5** includes running the checklist with both retrievers, the WRDS checks, and the cited-answer check (step 5.6).

**Use GPU time in parallel.** Question generation and training run on Colab while you label questions or write laptop code. That overlap is what makes the core fit.

**The core has almost no slack.** If time runs short, cut in this order:
1. The optional items (4.3, 2.3).
2. The whole-10-K method's 300-question synthetic sample (3.5); keep it on the hand-written set only.
3. Objective 5's case studies, then its WRDS checks.

Keep objectives 1, 2, 3 and 5, objective 4's breakdowns and the write-up. They answer both "does it work?" and "what is it for?"

---

## Appendix: fallback on the school GPU (Windows)

Use this only if Colab isn't available.
- **The machine:** the school computer runs Windows, you don't have admin rights, and Python 3.15.2 is already installed. These steps make PyTorch run on its GPU, an NVIDIA RTX 4000 SFF Ada with 20 GB.
- **Packages:** almost every package the project needs has Windows builds for Python 3.15 (checked Nov 19), including PyTorch, bitsandbytes, pyarrow and sentencepiece.
- **The exception is `ranx`.** Its dependencies `lz4` and `fastparquet` have no 3.15 builds (Nov 24), and without admin rights you can't install the compiler they'd need. Leave `ranx` out there and score on the laptop.

**A. Check the driver.**
- Run `nvidia-smi`. It should list the NVIDIA RTX 4000 SFF Ada Generation and show "CUDA Version: 13.x" at the top.
- You don't need to install the CUDA toolkit. PyTorch's GPU builds bring their own CUDA libraries; only the driver (596.71) has to be new enough.

**B. Install `uv` for your user only:** `py -3.15 -m pip install --user uv`. It creates the environments and installs packages, and nothing needs admin rights.

**C. Set user environment variables.** Open Windows Settings and search for "Edit environment variables for your account"; this needs no admin rights.
- `HF_HOME`: a folder with about 60 GB free, for downloaded models (Mistral-7B, Qwen2.5-14B, NV-Embed-v2). Without admin rights, Hugging Face stores full copies on Windows instead of links, so it needs the extra space.
- `HF_TOKEN`: your Hugging Face token.
- `SEC_IDENTITY`: your name and email, for SEC requests.
- `PYTHONUTF8` = `1`: makes Python read and write files as UTF-8. Python 3.15 already does this by default; the variable keeps it that way for every tool. 10-K text is full of characters like "—" and "’", which break Windows' older default encoding.
- Also check whether the school computer keeps your files between logins. Keep the project, `HF_HOME` and the training checkpoints on a drive that does.

**D. Create this repo's environment.** In this repo's folder, in PowerShell:

```
uv venv --python 3.15
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
Get-Content requirements.txt | Where-Object { $_ -notmatch '^\s*ranx' } | Set-Content $env:TEMP\req-no-ranx.txt
uv pip install -r $env:TEMP\req-no-ranx.txt
```

- The `--index-url` matters: a plain install of torch on Windows gets the CPU-only build.
- `cu130` is PyTorch's CUDA 13.0 build, which gives torch 2.14.1.
- The last two lines install everything except `ranx`, which can't install on 3.15 there.

**E. Check the GPU.**

```
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -m bitsandbytes
```

- The first line should print `2.14.1+cu130 True NVIDIA RTX 4000 SFF Ada Generation`.
- If it says `+cpu` or `False`, the CPU build got installed. Run `uv pip uninstall torch` and repeat the torch line from step D.
- The second command should report that bitsandbytes found CUDA.

**F. Change your copy of the official code for Windows.** Clone your private copy (setup step F) next to this repo, then:
1. In it, run `uv python pin 3.15`.
2. Add these lines to its `pyproject.toml`. The `bitsandbytes` dependency is already there from setup step F. These lines only affect `uv`; Colab's `pip` ignores them.

   ```
   [[tool.uv.index]]
   name = "pytorch-cu130"
   url = "https://download.pytorch.org/whl/cu130"
   explicit = true

   [tool.uv.sources]
   torch = [{ index = "pytorch-cu130" }]
   ```

3. In `src/stair/train.py`, when `num_gpus` is 1, start the trainer with plain Python (`sys.executable`) instead of `torchrun`. One GPU needs no distributed setup, and this avoids two things Windows lacks: NCCL and torchrun's libuv rendezvous.
4. Repeat check E inside it.
5. Don't turn on `torch.compile`. It needs Triton, which isn't available on Windows. Attention uses PyTorch's built-in SDPA.

**Done when:** the bundled example trains and evaluates on the school GPU, and `nvidia-smi` shows GPU memory in use during training.
