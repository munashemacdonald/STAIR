# How to complete each objective

This is the companion to `STAIR_Project_Overview.pdf`. For each objective it gives:
- what the objective answers
- the steps to complete it
- what you will have when it's done

Time estimates are in 8-hour working days and are summed up in the **Time** section at the end. `PLAN.md` has the same work as a checklist.

---

## Before you start (about 1 day)

1. **Sector and companies (done, Nov 17).**
   - Technology: 8 training companies, plus 2 held-out first-time filers (CRWV, FIG). They're listed in `config.yaml`.
   - Fiscal years differ by company, so each company's own FY2024 and FY2025 are used. The held-out companies have FY2025 only.
2. **Set up the GPU machine** by following "Setting up the GPU machine (Windows)" below.
3. **Set up the project folder** so every later step has a place to write:

   ```
   data/raw/          downloaded 10-K HTML, one folder per company-year
   data/parsed/       toc.json + content.jsonl per filing
   data/questions/    train / dev / test files
   src/               parser, question generation, retrievers, training, scoring
   checkpoints/       LoRA adapters
   results/           result tables (CSV) and figures
   ```

4. **Keep one config file** (`config.yaml`) with the company list, fiscal years, model names and random seed. Every number in the write-up should be reproducible from it.

## Setting up the GPU machine (Windows)

The school computer runs Windows, you don't have admin rights, and Python 3.15.2 is already installed. These steps make PyTorch run on the GPU.

Every package the project needs has Windows builds for Python 3.15 (checked Nov 19), including PyTorch, bitsandbytes, pyarrow and sentencepiece.

**A. Check the driver.**
- Run `nvidia-smi`. It should list the NVIDIA RTX 4000 SFF Ada Generation and show "CUDA Version: 13.x" at the top.
- You don't need to install the CUDA toolkit. PyTorch's GPU builds bring their own CUDA libraries; only the driver (596.71) has to be new enough.

**B. Install `uv` for your user only:** `py -3.15 -m pip install --user uv`. It creates the environments and installs packages, and nothing needs admin rights.

**C. Set user environment variables.** Open Windows Settings and search for "Edit environment variables for your account"; this needs no admin rights.
- `HF_HOME`: a folder with about 60 GB free, for downloaded models (Mistral-7B, Qwen2.5-14B, NV-Embed-v2). Without admin rights, Hugging Face stores full copies on Windows instead of links, so it needs the extra space.
- `HF_TOKEN`: your Hugging Face token.
- `SEC_IDENTITY`: your name and email, for SEC requests.
- `PYTHONUTF8` = `1`: makes Python read and write files as UTF-8. Python 3.15 already does this by default; the variable keeps it that way for every tool. 10-K text is full of characters like "—" and "’", which break Windows' older default encoding.
- Also check whether the school computer keeps your files between logins. Keep the project and `HF_HOME` on a drive that does.

**D. Create this repo's environment.** In this repo's folder:

```
uv venv --python 3.15
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
```

- The `--index-url` matters: a plain install of torch on Windows gets the CPU-only build.
- `cu130` is PyTorch's CUDA 13.0 build, which gives torch 2.14.1.

**E. Check the GPU.**

```
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -m bitsandbytes
```

- The first line should print `2.14.1+cu130 True NVIDIA RTX 4000 SFF Ada Generation`.
- If it says `+cpu` or `False`, the CPU build got installed. Run `uv pip uninstall torch` and repeat the torch line from step D.
- The second command should report that bitsandbytes found CUDA.

**F. Set up your fork of the official code.**
1. Fork [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean) on GitHub, and clone your fork next to this repo.
2. In the fork, run `uv python pin 3.15`.
3. Add these lines to the fork's `pyproject.toml`, and add `"bitsandbytes"` to its `dependencies` list:

   ```
   [[tool.uv.index]]
   name = "pytorch-cu130"
   url = "https://download.pytorch.org/whl/cu130"
   explicit = true

   [tool.uv.sources]
   torch = [{ index = "pytorch-cu130" }]
   ```

4. In `src/stair/train.py`, when `num_gpus` is 1, start the trainer with plain Python (`sys.executable`) instead of `torchrun`. One GPU needs no distributed setup, and this avoids two things Windows lacks: NCCL and torchrun's libuv rendezvous.
5. Repeat check E inside the fork.
6. Don't turn on `torch.compile`. It needs Triton, which isn't available on Windows. Attention uses PyTorch's built-in SDPA.

**Done when:** the fork's bundled example trains and evaluates (objective 2, step 2.1), and `nvidia-smi` shows GPU memory in use during training.

## How the objectives depend on each other

- **1 (data)** comes first. Everything else uses it.
- **2 (train STAIR)** needs 1.
- **3 (compare)** needs 1 and 2.
- **4 (new filings)** needs the FY2024 half of 1 and the training recipe from 2.
- **5 (investment use)** needs both years from 1, plus 2. It does not need 3 or 4.
- **6 (scale)** reruns 1–4 on more companies.

---

## Objective 1: Build a 10-K question dataset (5–6 days)

**What it answers:** nothing on its own. It builds the material every other objective uses. It is also where most of the risk sits: 10-K headings aren't marked up consistently, so the parser needs the most care.

### Steps

**1.1 Download the filings.**
- Use `edgartools` to download each company's FY2024 and FY2025 10-K from EDGAR. Skip 10-K/A amendments.
- Set your identity (name and email) first, because the SEC requires it, and stay under 10 requests per second.
- Save the main HTML document to `data/raw/<TICKER>_<FY>/`.
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
- Budget 20–30 minutes per filing, so roughly a day for all 18.
- Note how many fixes each filing needed. That is the parser's error rate, which goes in the write-up and matters for objective 6.

**1.7 Generate questions with an LLM.**
- Split each leaf section into paragraphs of about 100–300 words.
- For each paragraph, have an open instruction model write 3 questions that the paragraph answers. The paper used Mixtral 8x7B, which doesn't fit in 20 GB. Use Qwen2.5-14B-Instruct in 4-bit on the GPU (see `config.yaml`) or a larger model through the Hugging Face API.
- The official code's question generator works with any OpenAI-compatible endpoint, so it can run through the Hugging Face API as it is.
- In the prompt, ask for questions an analyst would ask, without reusing the heading's words. Otherwise the task turns into keyword matching.
- Label each question with its paragraph's section id.

**1.8 Filter the questions.**
- Remove exact duplicates and near-duplicates (embedding cosine similarity above about 0.9).
- Have the LLM check each question against its paragraph ("Can this question be answered from this paragraph?") and drop the ones it says no to.
- Read 100 random questions yourself and count how many are good. If fewer than about 80% are, fix the prompt and regenerate.

**1.9 Build the hand-written test set from FinanceBench's analyst questions.**
- **Where the questions come from:** FinanceBench (Islam et al., 2023) has 25 generic questions, written with financial analysts, that it asks of every company (its "domain-relevant questions", dg01–dg25). They're saved as templates in `data/questions/financebench_templates.jsonl`.
- **What to ask:** all 25, of every company's FY2025 filing, held-out companies included. That's up to 250 questions.
- **What to skip:** any template that doesn't apply, e.g. inventory turnover for a software company. Note why you skipped it.
- **How to label:** mark every section that answers each question, not just one. Several templates need more than one, e.g. "What drove operating margin change?" uses both MD&A and the income statement.
- **Label before you look:** label each question before you see any method's output.
- **dg04 (registered debt securities):** this is answered on the 10-K cover page. Keep the cover page as a section, or skip dg04.
- **Optional extra questions:** add your own company-specific questions, following FinanceBench's rules for its annotator-written questions: realistic, varied, and needing some reasoning.
- **Why this set matters most:** these are your most realistic test. Analysts wrote them without seeing your tables of contents, while the synthetic questions are easier and share the training questions' style.
- **In the write-up:** describe this test set, in the methods section, as FinanceBench's 25 domain-relevant questions (Islam et al., 2023), adapted to each company's FY2025 10-K, with section labels you added. Say how many you skipped and why. That's what lets readers recognise and trust the test set.
- **Licence:** CC BY-NC 4.0. Credit FinanceBench, and keep any question set you publish from them non-commercial.
- **Time:** labelling about 250 questions takes roughly 1–1.5 days.

**1.10 Split into train, dev and test.**
- **Held-out companies:** all their questions go into `test_heldout.jsonl`. None are used for training.
- **The other 8 companies:** split 80 / 10 / 10 into train, dev and test, by paragraph rather than by question. Questions written from the same paragraph then never land on both sides of the split.
- **Your hand-written questions:** these go into `test_handwritten.jsonl` and are never used for training.
- Do this separately for FY2025 and FY2024, because objective 4 trains on FY2024.

### Result when complete

- **Parsed filings:** 18 checked filings in `data/parsed/` (8 companies × 2 years, plus FY2025 for the 2 held-out companies), each with `toc.json` and `content.jsonl`.
- **Question files, for each year:**
  - `train`
  - `dev`
  - `test_synthetic`
  - `test_handwritten` (up to 250 questions from FinanceBench's 25 templates, FY2025)
  - `test_heldout`
- **Parser:** `src/parse_10k.py`, which turns any 10-K into those two files with one command.
- **Dataset table** for the write-up:

| Company | Year | Leaf sections | ToC length (tokens) | Train | Dev | Test | Parser fixes needed |
|---|---|---|---|---|---|---|---|

---

## Objective 2: Train STAIR on the filings (3–4 days, much of it GPU time)

**What it answers:** does STAIR, run with the authors' own code, work on 10-K filings?

**Build on the official code.** The authors' code is public: [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean), commit `3729c54`. Build on it instead of writing training code from scratch.
- In the write-up, call it the "official implementation", not a "baseline".
- The repo has no licence file. Keep it in your own fork until the authors give permission, and list every change you make.

### Steps

**2.1 Run the official code with only the Windows changes.**
- Set up the fork as in "Setting up the GPU machine (Windows)", step F: the CUDA build, `bitsandbytes`, and no `torchrun`. Change nothing else yet.
- The example's `prepare-data` step writes questions through an OpenAI-compatible LLM endpoint. Point it at the Hugging Face API with these variables:
  - `STAIR_LLM_API_BASE=https://router.huggingface.co/v1`
  - `STAIR_LLM_API_KEY`: your Hugging Face token
  - `STAIR_LLM_MODEL`: a model the API serves
- Run the bundled example (`prepare-data` → `train` → `eval`), and watch `nvidia-smi` during training to confirm the GPU is in use.

**2.2 Add 4-bit loading.**
- The code loads models in bf16 only. Mistral-7B-Instruct-v0.2 in bf16 takes about 14.5 GB before any training overhead, which is too tight for 20 GB.
- Load it in 4-bit (`bitsandbytes`, NF4) and keep the paper's LoRA settings (rank 16, alpha 32).

**2.3 Reproduce one SearchTome domain.**
- Train on the finance domain from Hugging Face (`meghanadh/SearchTome`) with Mistral-7B and the settings you'll use for 10-Ks.
- Compare your R@1 with the paper's 82.8 for finance. The public release has one book per domain while the paper used three, so expect some gap.
- If the 10-K results later look poor, this tells you whether the cause is your changes or the base code.

**2.4 Adapt the code for 10-Ks.**
- **Many filings per run:** the code reads one table of contents per dataset. Change the data loading so each question carries its filing's id and is paired with that filing's ToC, in training and evaluation.
- **Prompt:** replace "Book: {title}" with the company and filing, e.g. "NVIDIA Corp., Form 10-K, FY2025". Keep the rest of the prompt.
- **Input:** skip the PDF step. Your parser (objective 1) already writes the ToC in the code's format.
- **Length:** measure the longest ToC plus question in tokens, and raise `max_seq_length` (2,048 by default) above it.
- Run the repo's tests after each change.

**2.5 Time one epoch first, then set the schedule.**
- The paper allowed up to 200 epochs with early stopping. That won't fit your time.
- Time one epoch, then cap training at whatever fits overnight (likely 5–15 epochs).

**2.6 Pick the best checkpoint.**
- The code keeps the checkpoint with the lowest dev loss. The paper chose by dev Recall@1.
- Switch to dev Recall@1 if time allows. Either way, say in the write-up which one you used.

**2.7 Measure invalid outputs, then add the restriction as your extension.**
- **Official evaluation:** the code generates freely (greedy) and counts titles that aren't in the ToC afterwards. Keep that as the faithful replication, and report its invalid-output rate (the paper reported 0.05%).
- **Your extension:** for each filing, build a trie of its leaf titles' token sequences and pass it to `generate` through `prefix_allowed_tokens_fn`. The model can then only produce real titles. For a top 3, use beam search with 3–5 beams.
- Report both versions. With the restriction on, invalid outputs are zero by design.

### Result when complete

- **Model:** the best LoRA adapter, in `checkpoints/stair_fy2025/`.
- **Reproduction check:** your SearchTome finance R@1 next to the paper's.
- **Code:** your fork of the official code, with a list of every change, for the write-up.
- **Figure:** the dev metric over training.
- **Retrieval script:** give it a company and a question, and it returns the top 3 sections.
- **Recorded numbers:**
  - dev R@1
  - invalid-output rate, unrestricted and restricted
  - training time
  - GPU memory used

---

## Objective 3: Compare STAIR with the alternatives (2 days)

**What it answers:** the main research question. Does STAIR find the right 10-K section more often than standard RAG retrieval, and at what cost?

None of the comparison methods is trained; only STAIR is fine-tuned (decided Nov 9). The paper's STAIR vs. DSI result is cited, not reproduced: 82.6 vs. 76.9 average R@1 on SearchTome, with invalid outputs of 0.05% vs. 3.25%.

### Steps

**3.1 Write the scoring script first.**
- Every method produces the same output: for each question, a ranked list of section ids.
- The script compares that list with the correct section and reports R@1, R@3 and nDCG@3, using `ranx`.
- With one correct section per question, R@1 is simply the share of questions where the top answer is right.
- Test the script on a made-up run whose scores you already know.
- Score every method with this one script, STAIR included. The official code's eval reports the same metrics, so check the two agree.

**3.2 BM25 (keyword RAG).**
- Index each leaf section's text with `rank_bm25`, with a separate index per filing.
- Search only the filing of the company the question is about.

**3.3 Embedding retrieval (dense RAG).**
- Cut long sections into chunks of about 300 words and embed them. NV-Embed-v2 matches the paper; `bge-large-en-v1.5` is a lighter option.
- Score each section by its best-matching chunk.

**3.4 LLM given the ToC, without fine-tuning.**
- Send the STAIR prompt to the untuned Mistral. The paper got 13.8% R@1 this way.
- Send it to one stronger, 70B-class model through the Hugging Face API as well. The paper used Claude Haiku 4.5.
- Count invalid outputs for both.

**3.5 LLM given the whole 10-K.**
- Send the full filing text, with each section marked by its id and title, and ask which section answers the question.
- A 10-K is often 50,000–100,000 tokens. This needs a long-context model through the Hugging Face API, and it's the slowest and most expensive method. Estimate the token cost before running it.
- Run it only on the hand-written questions plus a random sample of about 300 synthetic ones.

**3.6 Significance test.**
- For STAIR against each other method, run a paired randomization test on per-question R@1 (Smucker et al., 2007).
- `ranx` has it built in as Fisher's randomization test.
- Report the p-values.

**3.7 Cost and speed.** For each method, record seconds per question and cost per 1,000 questions (GPU hours or API charges).

**3.8 Break down the results**
- by test set: synthetic, hand-written and held-out companies
- by Item: 1, 1A, 7 and 8

This shows where STAIR helps and where it doesn't.

### Result when complete

The main results table of the project:

| Method | R@1 synthetic | R@1 hand-written | R@1 held-out companies | R@3 | nDCG@3 | Invalid outputs | Sec / question | Cost / 1k questions | p vs. STAIR |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | | | | | | – | | | |
| Embeddings (dense) | | | | | | – | | | |
| Mistral + ToC, untuned | | | | | | | | | |
| Stronger LLM + ToC | | | | | | | | | |
| LLM + whole 10-K | | | | | | | | | |
| **STAIR** | | | | | | | | | – |

It also gives a one-sentence answer to the main question, for example: "On 10-K filings, STAIR's R@1 is X points above the best RAG retriever (p = …)." If STAIR doesn't win, that is still a valid result, and the breakdown in 3.8 explains why.

---

## Objective 4: Test whether it holds up on new filings (1.5–2 days)

**What it answers:** each company files a new 10-K every year. Does a model trained on last year's filings still work on this year's without retraining?

### Steps

**4.1 Train STAIR on FY2024.** Use the same recipe as objective 2, on the FY2024 training questions of the 8 training companies.

**4.2 Test it on FY2025.**
- Give it the FY2025 test questions with the FY2025 ToC in the prompt, and restrict its output to FY2025 titles.
- Compare it with the FY2025-trained STAIR from objective 2 on the same questions. The gap is the cost of not retraining.

**4.3 Measure how much the ToCs changed.**
- Match each FY2025 title against the FY2024 ToC: exact match first, then fuzzy match for small renames.
- Label each FY2025 section as unchanged, renamed or new.

**4.4 Break down accuracy by that label.** Expect the drop to be concentrated in new and renamed sections.

**4.5 Bring in the held-out companies.** Objective 3 already measured R@1 on companies STAIR never trained on, the other kind of "new filing". Report it here as well.

### Result when complete

| Model | Tested on | R@1 all | R@1 unchanged sections | R@1 renamed | R@1 new |
|---|---|---|---|---|---|
| STAIR trained on FY2025 | FY2025 test | | | | |
| STAIR trained on FY2024 | FY2025 test | | | | |
| BM25 (re-indexed, for reference) | FY2025 test | | | | |
| STAIR trained on FY2025 | held-out companies | | | | |

It also gives a practical conclusion, for example: "Without retraining, STAIR loses X points, mostly on new sections, so it should be retrained once each filing season."

---

## Objective 5: Show the investment use (2 days)

**What it answers:** what an analyst actually gets from STAIR. That is consistent, cited answers to the same questions across many companies and years.

### Steps

**5.1 Write the checklist, with a scoring scale for each question.**

Write 10–15 due-diligence questions, for example:
- How dependent is the company on its largest customers?
- What are the main supply-chain risks?
- How much debt matures in the next two years?
- Did management or the auditor report a material weakness in internal controls?
- Is there significant pending litigation?
- Did any accounting policy change this year?
- How cautious is management about demand and the outlook?
- Were there material cybersecurity incidents (Item 1C)?

Several of FinanceBench's templates fit here too, for example dg09 (revenue concentration), dg11 (litigation), dg20 (customers) and dg22 (debt). Reusing them links the checklist to the test set from step 1.9.

For each question, write a 1–5 scale in plain words. For customer dependence, for example: 1 = no customer above 10% of revenue … 5 = one customer above 30%. The written scale is what makes scores comparable across companies and repeatable.

**5.2 Run the checklist for every company and year.**
- STAIR picks the section.
- An answering LLM reads that section's text and returns JSON with:
  - a 2–3 sentence answer
  - the score on the scale
  - a supporting quote
  - the section id
- Use temperature 0.

**5.3 Add the year-over-year change.**
- Score change = FY2025 score − FY2024 score.
- For each question, also have the LLM read the FY2024 and FY2025 sections side by side and summarise what's new.

**5.4 Check the quality.**
- **Repeatable:** re-run each question reworded. Report how often the score stays the same or moves by at most 1.
- **Sensible:** before looking at the LLM's scores, score about 20 (company, question) pairs yourself on the same scale. Report exact agreement and agreement within 1 point.
- **Right section:** for the same 20 pairs, check that STAIR picked the right section.
- **Spread:** show how each question's scores are distributed. If every company gets the same score, that question can't tell companies apart.

**5.5 Write 2–3 short case studies.** Take the biggest score changes, show the FY2024 and FY2025 text side by side, and explain what changed.

### Result when complete

- **Signal table:** `results/signal_table.csv`, with one row per company × question. Each row has:
  - FY2024 score
  - FY2025 score
  - the change
  - the section each score came from
  - the supporting quotes
- **Heat map:** companies down the side, questions across the top, colour = FY2025 score, arrows for changes.
- **Quality numbers,** for example: "Scores agreed with mine on 15 of 20 pairs exactly, and 19 of 20 within one point."
- **Case studies:** 2–3 of them.
- **Scope sentence:** a clear statement that this shows how the signals are built, not that they predict returns.

---

## Objective 6 (stretch): Scale to 25–30 companies (3 days)

**What it answers:** when the number of companies triples, does the approach still work, and does the parser still work without hand-fixing?

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

| Part | Working days (8 h) |
|---|---|
| Before you start | 1 |
| Objective 1 | 5–6 |
| Objective 2 | 3–4 |
| Objective 3 | 2 |
| Objective 4 | 1.5–2 |
| Objective 5 | 2 |
| Objective 6 (stretch) | 3 |
| Write-up | 3 |
| **Total** | **20.5–23** |

At 40 hours a week, that is about 4–4.5 weeks. Objective 2 includes setting up the official code and reproducing one SearchTome domain. Objective 1 includes labelling the FinanceBench test questions.

If time runs short, cut in this order:
1. Objective 6.
2. Objective 4.

Keep objectives 1, 2, 3 and 5 and the write-up. They answer both "does it work?" and "what is it for?"
