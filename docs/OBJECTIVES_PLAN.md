# How to complete each objective

This is the companion to `STAIR_Project_Overview.pdf`. For each objective it gives:
- what the objective answers
- the steps to complete it
- what you will have when it's done

Time estimates are in 8-hour working days and are summed up in the **Time** section at the end. `PLAN.md` has the same work as a checklist.

## What this project tests

**The paper's question, and the one it left open.**
- The STAIR paper trains a model **per book**, for up to 200 epochs, and tests it on held-out questions **from the same book** (sections 4.2 and 5.2).
- So STAIR learns two things: the book's content, stored in its weights, and how to navigate the book's table of contents. Its 82.6% R@1 relies on both.
- Section 7 leaves **"a complete zero shot setup"**, meaning STAIR on documents it never trained on, as future work.

**The core test, in one sentence:** does STAIR, trained on FY2024 tables of contents, find the right section of an FY2025 10-K more often than the best standard RAG pipeline (hybrid + reranker)? It's measured at two levels:
- **Temporal zero-shot** (`seen_company`): a new filing from a company seen in training.
- **Cross-company zero-shot** (`first_time` + `established`): a new company and a new filing.

Objective 5 then shows what STAIR's cited answers are good for: a due-diligence checklist across companies.

**Why 10-Ks.** Their Items and accounting notes follow a legally fixed structure. If a zero-shot table-of-contents retriever works anywhere, it should work here; if it fails here, that marks a limit of the approach.

**What not to compare.** The paper's 82.6% was measured inside the corpus STAIR trained on, against weaker baselines. It isn't comparable with these results.

**Both outcomes are results:**
- **If STAIR matches or beats hybrid + reranker,** training on tables of contents transfers to new filings and companies without retraining.
- **If hybrid + reranker wins,** the paper's gains depend on having memorized the document, and live filings need their text read.

## What's in, and what's left out (Nov 25)

The plan was cut to the core test so that it fits the 15 days left on Nov 25.

**In:**
- **One version of STAIR:** 5-beam search, no constraint on its output.
- **Five comparison methods,** none of them trained: BM25, embeddings, their hybrid, hybrid + reranker (the row to beat), and the untuned Mistral given the table of contents.
- **Hand-written headline test:** about 280 questions across 16 companies.
- **Synthetic tests:** generated questions on the FY2025 filings.
- **Statistics:** two co-primary comparisons.
- **New filings:** a breakdown of results by whether a section is new, renamed or unchanged.
- **Investment use:** a simple checklist demonstration.

**Left out, and listed as future work in the write-up:**

| Left out | Why it isn't needed for the core test |
|---|---|
| Restricting STAIR's output with a trie, and a separate greedy version of STAIR | One version answers the question. Invalid outputs are still counted |
| A stronger 70B-class LLM, and an LLM given the whole 10-K, both through a paid API | Neither is standard RAG; they'd add cost, API setup and extra analysis |
| Checking generated questions for other correct sections before the methods run | Pooling after the methods run (step 3.7) finds them for every method equally |
| Breakdowns by Item, template group and number of correct sections; nDCG | The breakdown by company group answers the question |
| Sorting STAIR's misses into routing errors and vague headings | An explanation, not part of the test |
| A STAIR model trained on FY2025 (the paper's own setup) | The test is about filings STAIR never trained on |
| Reproducing a SearchTome domain | The bundled example already shows the official code runs |
| In objective 5: a second pipeline with hybrid + reranker, WRDS checks, rewording checks, case studies, a separate 50-question cited-answer check, a demo | Objective 5 demonstrates the use; a hand check of 20 answers is enough for that |
| Objective 6 (25–30 companies), XBRL figures, 10-Qs, an interface | After this project |

**Kept from the Nov 25 reviews:**
- STAIR trains on FY2024 only
- questions are generated per section
- the hand-written set is about 15 templates per company plus company-specific questions (about 280 in all since the four extra test companies were added on Nov 26)
- every RAG method uses chunks scored by their best chunk
- no company names in prompts or questions
- two co-primary comparisons
- at most 5 epochs, with the checkpoint chosen by dev R@1
- a 3,000-token cap per section in objective 5
- a time-box on parsing

---

## Before you start (about half a day left)

1. **Sector and companies (done, Nov 17).**
   - Technology: 8 training companies, 2 held-out first-time filers (CRWV, FIG), and 6 established test-only companies (ORCL, AMD, FFIV, AKAM, SWKS, JKHY). They're listed in `config.yaml`.
   - The test-only companies separate "a company the model never saw" from "a company's first 10-K". All have years of filings; none is trained on. ORCL and AMD were added on Nov 23.
   - **FFIV, AKAM, SWKS and JKHY were added on Nov 26** to make the cross-company result cover more firms: networking, cloud security, chips and bank software. They were chosen by the author, not drawn at random, before any results existed and before their filings were downloaded. Say so in the write-up.
   - Fiscal years differ by company, so each company's own FY2024 and FY2025 are used. The held-out and test-only companies use FY2025 only.
   - **STAIR trains on the 8 training companies' FY2024 filings only, and every test uses FY2025.** Each FY2025 test question has a `group`: `seen_company` (the 8, in a new year), `first_time` (CRWV, FIG) or `established` (ORCL, AMD, FFIV, AKAM, SWKS, JKHY).
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

**Where each step runs** (`runs_on` in `config.yaml`). No paid API is used.

| Where | What runs there |
|---|---|
| Laptop (CPU, `.venv`) | downloading and parsing (1.1–1.6); the hand-written questions and their labels (1.9); BM25 (3.2); merging the hybrid's rankings (3.4); checking the judge's pooled decisions (3.7); scoring and statistics (3.1, 3.8–3.10); objective 4; the write-up |
| Colab GPU | question writing and the pooling judge (1.7–1.8, 3.7), with vLLM; STAIR training and inference (objective 2); NV-Embed-v2 and the reranker (3.3, 3.5); the untuned Mistral (3.6); objective 5's answering model (5.2) |

**How work moves between the laptop and Colab.** Every step is a script in `src/` that reads files and writes files. No step depends on what's held in a notebook's memory, so no notebook ever switches between the laptop and Colab.
- **Code** goes through GitHub: push from the laptop, pull in Colab.
- **Data** goes through Google Drive. The Drive desktop app syncs `My Drive/stair/` on the laptop, and Colab sees the same folder at `/content/drive/MyDrive/stair/`.
- **GPU work comes in about three sessions:**
  1. question writing
  2. training
  3. every GPU method on every test set, then the pooling judge and objective 5's answers

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
- **Never put tokens there.**
- **Models don't go there either.** Colab downloads them from Hugging Face in each session.

**C. Colab.**
- **Units:** Google AI Pro includes about 200 Colab compute units a month. Open a notebook at colab.research.google.com with the same Google account, and check the balance under Runtime → View resources.
- **GPU:** choose it under Runtime → Change runtime type.
  - **L4:** most GPU work.
  - **A100:** training, when Colab offers one.
  - **T4:** quick tests only. It has no bf16, and NV-Embed-v2 doesn't fit.
- **Cost:**
  - An L4 burns about 1.7 units an hour and an A100 about 5.4 (third-party measurements, March 2026). The project should need well under 100 units.
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
- **4 (new filings)** breaks down 3's results.
- **5 (investment use)** needs both years from 1, plus 2. It does not need 3 or 4.

## Licences

Checked Nov 23. Say in the write-up what each part is licensed under.

| What | Licence | What it means here |
|---|---|---|
| 10-K filings (SEC EDGAR) | Public | Free to use, and to publish what you build from them |
| Official STAIR code | None given | Use it in your own private copy, and ask before publishing changes (requested Nov 19). A GitHub fork would be public, so don't use one. Keep your changes as a separate module or patch, so this repo holds only your own code |
| FinanceBench templates | CC BY-NC 4.0 | Credit FinanceBench. Non-commercial only |
| Mistral-7B-Instruct-v0.2 | Apache 2.0 | Credit only |
| Qwen2.5-14B-Instruct and its AWQ version (writes the questions, judges, answers objective 5) | Apache 2.0 | Its outputs can be used to train STAIR |
| NV-Embed-v2 | CC BY-NC 4.0 | Non-commercial only |
| bge-reranker-v2-m3 | Apache 2.0 | Credit only |

**Overall:** FinanceBench and NV-Embed-v2 make the project non-commercial. That's fine for a university project; say so in the write-up.

---

## Objective 1: Build a 10-K question dataset (about 6 days)

**What it answers:** nothing on its own. It builds the material every other objective uses. It is also where most of the risk sits: 10-K headings aren't marked up consistently, so the parser needs the most care.

### Steps

**1.1 Download the filings.**
- Use `edgartools` to download each company's FY2024 and FY2025 10-K from EDGAR. Skip 10-K/A amendments.
- Set your identity (name and email) first, because the SEC requires it, and stay under 10 requests per second.
- Save the main HTML document to `data/raw/<TICKER>_<FY>/`.
- For the held-out (CRWV, FIG) and test-only (ORCL, AMD, FFIV, AKAM, SWKS, JKHY) companies, download the FY2025 10-K only. That makes 24 filings.
- Check whether the company puts its financial statements in Exhibit 13 instead of Item 8 (the filing will say "incorporated by reference"). If it does, download that exhibit too.

**1.2 Split each filing into Items.**
- The Items that matter most are 1, 1A, 1B, 1C, 2, 3, 5, 7, 7A, 8 and 9A.
- `edgartools` can return the text of each Item. Check its output against the filing's own table of contents.
- Skip the cover page, the signatures and the exhibit index.

**1.3 Find the headings inside each Item.**
- 10-K HTML rarely uses real heading tags. Headings are usually short lines (under about 12 words) that stand alone as a paragraph and are entirely bold, underlined or in a larger font.
- In Item 8, the notes start with patterns like "Note 7 – Debt" or "7. Debt".
- Write rules for these with `BeautifulSoup`.
- Limit the outline to three levels: Item → heading → subheading.
- **Time-box this to about 1.5 days.** If headings inside Items still aren't reliable by then, fall back to Items plus numbered Notes (and the MD&A headings if they work), and say so in the write-up.

**1.4 Number every section and write `toc.json`.**
- Number each node by its position. Item 7's third heading becomes "7.3" and its first subheading "7.3.1". Item 1A's second heading becomes "1A.2".
- The model's target is the number plus the title, e.g. "7.3 Liquidity and Capital Resources".
- This copies the paper's numbered titles and solves a real problem: 10-Ks repeat headings like "Overview" in several Items, and numbering makes every target unique.
- Each node has `id`, `title`, `level`, `parent` and `is_leaf`.
- Also write the official code's format, so its training code can read your filings directly: `{"title": ..., "table_of_contents": [{"id", "section_num", "title"}]}`. Set `title` to "SEC Form 10-K filing" for every filing (see step 2.3).

**1.5 Write `content.jsonl`.**
- Write one line per leaf section, with `id`, `title` and `text`.
- Convert tables to plain-text rows (cells separated by " | ") instead of dropping them. A lot of 10-K information is in tables.

**1.6 Check each table of contents by hand.**
- Open the filing next to its `toc.json` and fix missed or false headings.
- Budget 20–30 minutes per filing, so roughly a day and a half for all 24.
- Note how many fixes each filing needed. That is the parser's error rate, which goes in the write-up.

**1.7 Generate questions with an LLM, per section.**
- **Write questions per leaf section, not per paragraph.** Per paragraph, long boilerplate sections would flood training and the volume would be too large to train on in time.
  - **Training: the 8 training companies' FY2024 filings.** Each leaf section gets 3 questions, plus 1 for every 500 words beyond the first 500, up to 6. Write each question from one 100–300-word passage, with the passages spread across the section. That should give about 5,000 training questions (`questions` in `config.yaml`); if the count is far off, adjust the numbers.
  - **Dev and test: every FY2025 filing.** Each leaf section gets 2 questions.
- **The model:** Qwen2.5-14B-Instruct, served with vLLM on a Colab L4 as `Qwen/Qwen2.5-14B-Instruct-AWQ` (the same model in 4-bit). vLLM batches requests and is much faster than `transformers`.
  - vLLM installs its own torch, which is fine because this session does nothing else.
  - If vLLM won't install, use `transformers` in 4-bit with batched generation.
  - The paper used Mixtral 8x7B, which needs about 25 GB even in 4-bit: too much for an L4.
- **The prompt:**
  - Ask for questions an analyst would ask, without reusing the heading's words. Otherwise the task turns into keyword matching.
  - **Never name the company:** ask the generator to write "the company" instead. Drop or rewrite any question that names it anyway.
- **Label each question with its section's id.** A training question always has exactly one target section.
- **Keep the hand-written test questions out of generation.**
  - Never use the FinanceBench or checklist templates, or any company-specific test question, as examples in the prompt.
  - After generating, embed every generated question and drop any whose cosine similarity to a hand-written test question is above about 0.85.
- **Storage:** write the questions to `My Drive/stair/questions/`, then copy them into `data/questions/` on the laptop and commit them.

**1.8 Filter the questions.**
- **Duplicates:** remove exact duplicates and near-duplicates (embedding cosine similarity above about 0.9).
- **Answerability:** have the LLM check each question against its passage ("Can this question be answered from this passage?") and drop the ones it says no to.
- **Quality:** read 100 random questions yourself and count how many are good. If fewer than about 80% are, fix the prompt and regenerate.
- **Other correct sections:** a 10-K repeats facts across Items, so a question can have more than one correct section. Those are found later, by pooling every method's answers (step 3.7), so that no method is favoured.

**1.9 Build the hand-written test set: the headline test.**
- **Where the questions come from:** three sources, all written without seeing your tables of contents.
  - **FinanceBench templates:** FinanceBench (Islam et al., 2023) has 25 generic questions, written with financial analysts, that it asks of every company (its "domain-relevant questions", dg01–dg25). They're saved as templates in `data/questions/financebench_templates.jsonl`.
    - **7 of them are used** (chosen Nov 25, before any table of contents was parsed; `handwritten` in `config.yaml`): dg09, dg11, dg20 and dg22, which objective 5 uses, plus dg14 (operating margin), dg19 (cash flows) and dg24 (working capital).
    - The other 18 mostly repeat these (quick ratio, several gross-margin variants), need the cover page (dg04), or fit few tech companies (dg25, inventory turnover).
  - **Checklist templates:** 8 analyst questions in `data/questions/checklist_templates.jsonl` (ck01–ck08), fixed before any table of contents was parsed. They cover supply chain, debt maturities, internal controls, accounting changes, outlook, cybersecurity and buybacks. Objective 5 reuses them.
  - **Company-specific questions:** 56 in all, from what an analyst would want to know (news, earnings calls), in `data/questions/company_specific.jsonl`.
    - **2 per seen company (16) and 5 per unseen company (40).** The extra questions strengthen the cross-company comparison.
    - **Each company's questions are written before its filing is downloaded or parsed,** so before you see its table of contents. The first 36 were committed on Nov 26 before the parser was built; the 20 for the four companies added that day were committed before their filings were downloaded.
    - Follow FinanceBench's rules for its annotator-written questions: realistic, varied, and needing some reasoning.
- **No question names the company.** Fill `{company}` in the templates with "the company", and write the company-specific questions the same way. The `company` field still records which filing a question is about.
- **What to ask:** all 15 templates (7 FinanceBench + 8 checklist) of every company's FY2025 filing, held-out and test-only companies included, plus the company-specific questions. That's up to 296 questions (15 × 16 + 56), about 280 after skips.
- **Each question records its `group`:** `seen_company`, `first_time` or `established`.
- **Independent units:** a template's company versions tend to rise or fall together, so they count as one piece of evidence.
  - The seen group has about 31 independent units (15 templates + 16 questions), and the unseen group about 55 (15 + 40).
  - Only large differences are likely to reach significance: roughly 20 points or more in the seen group, and roughly 15 in the unseen group. The synthetic held-out set adds power for the unseen group (step 3.8).
- **What to skip:** any question that doesn't apply, e.g. buybacks for a company without a buyback programme. Note why you skipped it.
- **The labelling rule** is in `data/questions/README.md`; stick to it. A section counts as correct in one of two ways:
  - **Answers on its own:** a reader could answer the question from this section alone. Several sections can each do this; any one of them is enough.
  - **Needed together:** the answer needs this section plus others, and none of them answers alone.
  - **Doesn't count:** cross-references ("see Note 12"), general boilerplate, and sections that only support an answer another section gives.
- **Label before you look:** label each question before you see any method's output.
- **Second labeller, before you label the rest:**
  - Have a classmate label about 30 questions, picked at random across templates and companies, with the same rule and without seeing your labels.
  - Compare, settle the disagreements, and tighten the rule where it was unclear. Then label the rest.
  - In the write-up, report how often your `answers_alone` sets match exactly, and how often they overlap.
  - If nobody is available, re-label the same 30 yourself at least three days later, without looking at your first labels.
- **Why this set is the headline:** it's the most realistic test, written without seeing your tables of contents. The synthetic questions share the training questions' style, so STAIR's synthetic scores are inflated.
- **Licence:** CC BY-NC 4.0. Credit FinanceBench, and keep any question set you publish from them non-commercial.
- **Time:** labelling about 280 questions takes roughly 2.5 days, plus a couple of hours for the second-labeller check. The company-specific questions are already written.

**1.10 Split into train, dev and test.**
- **Training, `train.jsonl`:** every FY2024 question of the 8 training companies. STAIR never trains on an FY2025 filing.
- **Dev and synthetic test:** the 8 training companies' FY2025 questions, split by section. 20% of sections go to `dev.jsonl` and 80% to `test_synthetic.jsonl`, so no section is in both. Dev comes from FY2025 so that every choice is made in the same setting as the test.
- **Held-out and test-only companies:** all their FY2025 questions go into `test_heldout.jsonl`.
- **Your hand-written questions:** these go into `test_handwritten.jsonl`.
- **Every FY2025 question** has a `group` field: `seen_company`, `first_time` or `established`.

### Result when complete

- **Parsed filings:** 24 checked filings in `data/parsed/` (8 companies × 2 years, plus FY2025 for the 8 unseen companies), each with `toc.json` and `content.jsonl`.
- **Question files:** `train` (FY2024, about 5,000), `dev` and `test_synthetic` (FY2025, the 8 training companies), `test_heldout` (FY2025, the 8 unseen companies), and `test_handwritten` (about 280, with a second-labeller agreement figure).
- **Parser:** `src/parse_10k.py`, which turns any 10-K into those two files with one command.
- **Dataset table** for the write-up:

| Company | Year | Leaf sections | ToC length (tokens) | Train | Dev | Test | Parser fixes needed |
|---|---|---|---|---|---|---|---|

---

## Objective 2: Train STAIR on the filings (about 1.5 days, much of it GPU time)

**What it answers:** does STAIR, run with the authors' own code, learn to pick 10-K sections? It trains on the 8 training companies' FY2024 filings only; objective 3 tests it on FY2025.

**Build on the official code.** The authors' code is public: [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean), commit `3729c54`.
- In the write-up, call it the "official implementation", not a "baseline".
- The repo has no licence file. Keep it in your own private copy until the authors give permission, and list every change you make.
- You edit the code on the laptop, push it, and pull it in the Colab notebook.

### Steps

**2.1 Run the official code unchanged, on Colab.**
- Set up your copy as in "Setting up the laptop and Colab", step F. Change nothing else yet.
- The example's `prepare-data` step writes questions through an OpenAI-compatible LLM endpoint. Point it at the Hugging Face API for this small example, with these variables set from Colab Secrets:
  - `STAIR_LLM_API_BASE=https://router.huggingface.co/v1`
  - `STAIR_LLM_API_KEY`: your Hugging Face token
  - `STAIR_LLM_MODEL`: a model the API serves
- Run the bundled example (`prepare-data` → `train` → `eval`) on an L4. Check Colab's resources panel during training to confirm the GPU is in use.

**2.2 Add 4-bit loading.**
- The code loads models in bf16 only. Mistral-7B-Instruct-v0.2 in bf16 takes about 14.5 GB before any training overhead, which is too tight for an L4's 24 GB with long table-of-contents prompts.
- Load it in 4-bit (`bitsandbytes`, NF4) and keep the paper's LoRA settings (rank 16, alpha 32).
- Use 4-bit on every GPU, A100 included, so every run uses the same setup whichever GPU Colab gives you.

**2.3 Adapt the code for 10-Ks.**
- **Many filings per run:** the code reads one table of contents per dataset. Change the data loading so each question carries its filing's id and is paired with that filing's ToC, in training and evaluation.
- **Prompt:** the code says "Book: {title}". Every filing's title is "SEC Form 10-K filing", with no company name or year (`prompt` in `config.yaml`). A company name in training would let the model learn "this company → its note layout" instead of reading the table of contents. Questions also say "the company" (steps 1.7 and 1.9). Keep the rest of the prompt.
- **Input:** skip the PDF step. Your parser (objective 1) already writes the ToC in the code's format.
- **Length:** measure the longest ToC plus question in tokens, and raise `max_seq_length` (2,048 by default) above it.
- Run the repo's tests after each change.

**2.4 Train, within a budget, and save often.**
- **Budget:** at most about 8 hours and **at most 5 epochs** per run (`training` in `config.yaml`).
  - The paper allowed 200 epochs so that STAIR would memorize each book. Here the test filings are ones it never saw, so long training would only memorize FY2024 and hurt FY2025. The official code's default is 3.
  - Every training example carries the filing's whole table of contents, likely 2,000–6,000 tokens. Estimate the time first from a 200-step timing: tokens per epoch (questions × average prompt length) ÷ tokens per second.
  - If the estimate is over budget, cut in this order: a two-level table of contents (Item → heading, relabelling questions to the level-2 parent), fewer epochs, and as a last resort a smaller model.
  - Compute units are hours × the GPU's rate: an 8-hour L4 run is about 14 units, and a 2-hour A100 run about 11.
- **Save often, and make training resumable.** Colab can stop a run partway through: the session time limit, an idle timeout, a closed tab or a sleeping laptop. Google AI Pro has no background execution, so keep the tab open and the laptop awake.
  - Save a checkpoint about every 30 minutes to `/content/drive/MyDrive/stair/checkpoints/<run>`, never to `/content`, which Colab wipes. From the 200-step timing, set `save_interval` to the steps that take about 30 minutes, and `eval_interval` to the same number or one that divides it.
  - **Keep every checkpoint** (`save_total_limit: null`), so each can be scored in step 2.5. Each is about half a gigabyte.
  - The official `train` command always starts from scratch. Add a `--resume` option that passes `resume_from_checkpoint=True` to the trainer, and test it once on the bundled example.
  - After an interruption, resume rather than restart, and note it in the training log.
- Record the estimated and actual training time, the GPU type and the units used.

**2.5 Pick the checkpoint by dev R@1.**
- The code keeps the checkpoint with the lowest dev loss, but loss can keep falling as the model gets better at the output format while it gets worse at picking sections. The paper chose by dev R@1.
- After training, run every checkpoint on the 300-question dev sample, with the test decoding (step 2.6), score R@1, and keep the best.
- Dev holds only seen companies, so the choice tunes for temporal zero-shot. Say so in the write-up.

**2.6 Decode with 5-beam search: the one version of STAIR.**
- **The output format** (checked in the code at `3729c54`): the prompt asks for "a Python list of strings", and the training completion is a newline followed by the list, for example `\n["7.3 Liquidity and Capital Resources"]`. The paper's Table 2 shows a single string instead; this project follows the code.
- **Decoding:** beam search with 5 beams and no constraint on the output.
  - **R@1** uses the top beam's section.
  - **The top 3** are the valid sections from the beams, in beam order, without repeats.
  - Greedy decoding would return a single section, so R@3 would just equal R@1. The paper reports a separate R@3 for STAIR, which needs a ranked list like this.
- **Invalid outputs:** count any title that isn't in the filing's table of contents and drop it from the ranking. The paper calls these hallucinations and reports 0.05%.

### Result when complete

- **Model:** the chosen LoRA adapter, trained on FY2024, in `checkpoints/stair_fy2024/` (on Drive).
- **Code:** your private copy of the official code, with a list of every change, for the write-up.
- **Figure:** dev R@1 for each checkpoint.
- **Retrieval script:** give it a filing and a question, and it returns the top 3 sections.
- **Recorded numbers:** dev R@1, the invalid-output rate, estimated and actual training time, the GPU type, GPU memory and units used.

---

## Objective 3: Compare STAIR with standard RAG (about 2 days)

**What it answers:** the main question. On FY2025 filings, does STAIR find the right section more often than standard RAG retrieval, for seen companies and for unseen ones?

None of the comparison methods is trained; only STAIR is (decided Nov 9). The RAG methods index the FY2025 filings directly, as a real system would. Say in the write-up that STAIR is the only method trained on these companies' filings, and that a reranker fine-tuned on the same questions is future work.

### Steps

**3.1 Write the scoring script first.**
- Every method produces the same output: for each question, a ranked list of section ids.
- **R@k:** a question counts as found at k if any of its correct sections is in the top k. This is "hit@k", the official code's `match@k`. "Correct" means `answers_alone` plus `pooled_answers_alone` (step 3.7). Also score without the pooled sections, and report both.
- **Needed-together questions:** leave them out of R@1 and R@3. Report, in one line, how often all their needed sections are in the top 3 (All@3).
- Use `ranx` where it fits, and test the script on a made-up run whose scores you already know.

**3.2 Chunks and BM25 (keyword RAG).**
- Cut every section into chunks of about 300 words, overlapping by 50, each starting with its section number and title. BM25, the embeddings and the reranker all use these chunks.
- Index the chunks with `rank_bm25`, with a separate index per filing, and search only the filing of the company the question is about.
- Score each section by its best chunk ("MaxP"), so long sections aren't penalised.

**3.3 Embedding retrieval (dense RAG).** Embed the chunks with NV-Embed-v2 (`bge-large-en-v1.5` if memory is tight), and score each section by its best-matching chunk.

**3.4 Hybrid: BM25 + embeddings, merged with reciprocal rank fusion.** Each section gets 1 / (60 + its rank) from each method; add the two and sort (Cormack et al., 2009). `ranx` does this with `fuse(..., method="rrf")`.

**3.5 Hybrid + reranker: the row to beat.**
- Run the cross-encoder `BAAI/bge-reranker-v2-m3` over the hybrid's top 20 sections, scoring every question–chunk pair (maximum 512 tokens). A section's score is its best chunk's score.
- A cross-encoder reads the question and the text together, which makes it the strongest standard pipeline that needs no training.
- Load it with `sentence-transformers` (`CrossEncoder`). Run it after NV-Embed-v2 has been unloaded.

**3.6 The untuned Mistral given the table of contents.**
- Give the untuned Mistral-7B-Instruct exactly STAIR's prompt and the same 5-beam decoding, on the same GPU type and 4-bit setup. The paper got 13.8% R@1 this way.
- This shows how much STAIR's fine-tuning adds to the same model. Count its invalid outputs too.

**3.7 Pool every method's answers, then rescore.**
- **Why:** a 10-K repeats facts across Items, so a method can pick a different section that also answers the question. Pooling finds those sections for every method equally; it's how TREC builds its labels.
- **When:** after every method has run on the test sets, and before any significance test.
- **What to pool:** each method's top 3 sections for each test question, minus any already labelled, one copy of each.
- **Who judges, without knowing which method found each section:** the question-writing model (Qwen2.5-14B, temperature 0), shown the question and the section's best-matching ~300-word chunk, asked "Does this text, on its own, answer the question?" For the hand-written set, it's given the labelling rule.
- **Check the judge:** for the hand-written set, check a random 50 of its decisions yourself, blind, and report how often you agree.
- **Record:** every "yes" goes in `pooled_answers_alone`, not in `answers_alone`, so the labels made before any method ran stay separate. Report how many pooled sections each method contributed.
- Don't pool the dev set.

**3.8 Statistics.**
- **Two co-primary comparisons, fixed before running any test** (`scoring.primary` in `config.yaml`). Both are STAIR against hybrid + reranker, on R@1, on the hand-written set:
  - **Temporal zero-shot:** the 8 seen companies' questions.
  - **Cross-company zero-shot:** the 8 unseen companies' questions.

  Judge each by the 95% confidence interval of the difference. Test each with a paired randomization test (Smucker et al., 2007; built into `ranx`), and correct the two p-values with Holm: the smaller must be below 0.025, the other below 0.05.
- **Confidence intervals for the hand-written set:** bootstrap the difference in R@1 (STAIR minus the other method), resampling whole units within each group: each template is one unit (all its company versions together), and each company-specific question is its own unit. Resampling single questions would make the results look more certain than they are.
- **Secondary comparisons:** STAIR against every other method, and both synthetic test sets, with Holm-adjusted p-values. Synthetic sets resample sections, not questions.
  - **The synthetic held-out set** (the 8 unseen companies pooled, about 2,400 questions) gets these full tests, because it gives cross-company zero-shot its statistical power. Say that it covers only these 8 companies.
- **Report the all-company hand-written R@1** as a descriptive number, without a test.
- **Small cells:** in any breakdown with fewer than about 30 questions, show the count and confidence interval, and draw no conclusion from it.
- **Freeze the test sets.** Make every choice (prompts, epochs, the checkpoint) on dev, never on test. Record the date of the first test run.

**3.9 Cost and speed.** For each method, record seconds per question with the GPU type, and Colab units per 1,000 questions. BM25 and the hybrid's merge run on the laptop and cost nothing.

**3.10 Break down by group:** seen companies, first-time filers and established companies, on the synthetic and hand-written sets. With only 2 first-time filers, don't claim a difference between first-time filers and established companies.

### Result when complete

The main results table:

| Method | R@1 hand-written, seen (co-primary) | R@1 hand-written, unseen (co-primary) | R@1 synthetic (seen / held-out) | R@3 | Invalid outputs | Sec / question | Units / 1k questions | p vs. STAIR (Holm-adjusted) |
|---|---|---|---|---|---|---|---|---|
| BM25 | | | | | – | | | |
| Embeddings (dense) | | | | | – | | | |
| Hybrid (BM25 + embeddings, RRF) | | | | | – | | | |
| Hybrid + reranker | | | | | – | | | |
| Mistral + ToC, untuned | | | | | | | | |
| **STAIR** (trained on FY2024) | | | | | | | | – |

R@1 and R@3 are hit@k, as defined in 3.1, after pooling; the scores before pooling go in an appendix table.

**The headline is the two hand-written columns,** each with the confidence interval of the difference from step 3.8. It gives a one-sentence answer for each level, for example: "On next year's filings of companies it trained on, STAIR's R@1 is X points above hybrid + reranker (95% CI …); on companies it never saw, the gap is Y points (95% CI …)." If STAIR doesn't win, that is still a valid result (see "What this project tests").

---

## Objective 4: Test whether it holds up on new filings (about a quarter of a day)

**What it answers:** each company files a new 10-K every year. Where does a model trained on last year's filings go wrong on this year's? Objective 3 already tests STAIR on FY2025; this objective breaks the seen companies' results down by what changed.

### Steps

**4.1 Measure how much the ToCs changed.** For each of the 8 training companies, match each FY2025 title against its FY2024 ToC: exact match first, then fuzzy match for small renames (`rapidfuzz`). Label each FY2025 section as unchanged, renamed or new.

**4.2 Break down accuracy by that label,** for the seen companies' FY2025 questions, synthetic and hand-written, showing the count in each column. Count FY2024 titles among STAIR's invalid outputs separately: they show the model falling back on last year's filing.

**4.3 Add the unseen companies** from objective 3, per company.

### Result when complete

| Model | Tested on | R@1 all | R@1 unchanged sections | R@1 renamed | R@1 new |
|---|---|---|---|---|---|
| STAIR | FY2025, seen companies | | | | |
| Hybrid + reranker (for reference) | FY2025, seen companies | | | | |

It also gives a practical conclusion, for example: "Trained on last year's filings, STAIR's errors concentrate in new sections, so it should be retrained each filing season."

---

## Objective 5: Show the investment use (about 1 day)

**What it answers:** what an analyst gets from STAIR: consistent, cited answers to the same questions across companies and years.

### Steps

**5.1 Write a scoring scale for each checklist question.**
- The checklist is fixed: the 8 checklist templates (ck01–ck08) and 4 FinanceBench templates (dg09 revenue concentration, dg11 litigation, dg20 customers, dg22 debt). All 12 are in the hand-written test set, so you already know how well STAIR retrieves the sections they need.
- For each, write a 1–5 scale in plain words. For customer dependence, for example: 1 = no customer above 10% of revenue … 5 = one customer above 30%.

**5.2 Run the checklist for every company and year with STAIR's top 3 sections.**
- An answering LLM (`objective5.answer_model`, temperature 0) reads the 3 sections and returns JSON with:
  - a 2–3 sentence answer
  - the score on the scale
  - a supporting quote
  - the id of the section the quote comes from
- **Cap each section at about 3,000 tokens.** If a section is longer, keep its best chunks for the question, picked by the reranker from step 3.5, up to the cap, in their original order. Record each prompt's length.
- STAIR was trained on the FY2024 filings, so for FY2024 it's searching filings it has seen. That's fine for building signals, but it isn't a test result.

**5.3 Add the year-over-year change:** score change = FY2025 score − FY2024 score, for the 8 companies with both years.

**5.4 Check 20 answers by hand.** For about 20 (company, question) pairs, without looking at the LLM's score first, record:
- **Score:** your own score on the same scale, and whether the LLM's matches exactly or within 1 point
- **Section:** whether STAIR's sections contained the answer
- **Quote:** whether the quote supports the answer
- **Citation:** whether the cited section exists and contains the quote

### Result when complete

- **Signal table:** `results/signal_table.csv`, with one row per company × question: FY2024 score, FY2025 score, the change, the section each score came from, and the supporting quotes.
- **Heat map:** companies down the side, questions across the top, colour = FY2025 score.
- **Quality numbers,** for example: "Scores agreed with mine on 15 of 20 pairs exactly, and 19 of 20 within one point; 18 of 20 citations were valid."
- **Scope sentence:** a clear statement that this shows how the signals are built, not that they predict returns.

---

## Time

Re-planned on Nov 25, with about 15 days left.

| Part | Working days (8 h) |
|---|---|
| Setup still to do (Colab, Drive, the official code, environment variables) | 0.5 |
| Objective 1 | 6 |
| Objective 2 | 1.5 |
| Objective 3 | 2 |
| Objective 4 | 0.25 |
| Objective 5 | 1 |
| Write-up | 2 |
| **Total** | **13.25** |

That leaves about 1.75 days of slack. Adding the four test companies on Nov 26 used about 1.5 days.
- **Objective 1** includes:
  - parsing, time-boxed to about 1.5 days, and checking 24 tables of contents by hand (about a day and a half)
  - labelling about 280 hand-written questions (about 2.5 days); the company-specific questions are already written
  - generating questions on Colab, mostly unattended GPU time
- **Use GPU time in parallel.** Question generation and training run on Colab while you label questions or write laptop code.

If time still runs short, cut objective 5's heat map and reduce its hand check to 10 answers. Never cut objectives 1–3 or the write-up.

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
