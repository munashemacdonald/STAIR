# How to complete each objective

This is the companion to `STAIR_Project_Overview.pdf`. For each objective it gives:
- what the objective answers
- the steps to complete it
- what you will have when it's done

Time estimates are in 8-hour working days and are summed up in the **Time** section at the end. `PLAN.md` has the original dated schedule, which assumed a Nov 2 start.

---

## Before you start (about 1 day)

1. **Pick the sector and 10 companies.**
   - Use companies whose fiscal year ends in December. Then "FY2024" and "FY2025" cover the same months for every company, and the FY2025 10-K (filed Feb–Mar 2026) is each company's latest. Check each fiscal year-end on the company's EDGAR page.
   - Avoid banks and insurers. Their 10-Ks are very long and organised differently.
   - Choose 2 of the 10 as **held-out companies** now, before you see any results.
2. **Set up the GPU machine.**
   - Install `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `datasets`, `vllm`, `sentence-transformers`, `rank_bm25`, `ranx`, `edgartools` and `beautifulsoup4`.
   - Confirm the setup with a 10-minute practice QLoRA run on a few hundred examples.
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

## How the objectives depend on each other

- **1 (data)** comes first. Everything else uses it.
- **2 (train STAIR)** needs 1.
- **3 (compare)** needs 1 and 2.
- **4 (new filings)** needs the FY2024 half of 1 and the training recipe from 2.
- **5 (investment use)** needs both years from 1, plus 2. It does not need 3 or 4.
- **6 (scale)** reruns 1–4 on more companies.

---

## Objective 1: Build a 10-K question dataset (4–5 days)

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
- Skip the cover page, the signatures and the exhibit index.

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

**1.5 Write `content.jsonl`.**
- Write one line per leaf section, with `id`, `title` and `text`.
- Convert tables to plain-text rows (cells separated by " | ") instead of dropping them. A lot of 10-K information is in tables.

**1.6 Check each table of contents by hand.**
- Open the filing next to its `toc.json` and fix missed or false headings.
- Budget 20–30 minutes per filing, so roughly a day for all 20.
- Note how many fixes each filing needed. That is the parser's error rate, which goes in the write-up and matters for objective 6.

**1.7 Generate questions with an LLM.**
- Split each leaf section into paragraphs of about 100–300 words.
- For each paragraph, have an open instruction model on the GPU write 3 questions that the paragraph answers. The paper used Mixtral 8x7B; a 7–8B model is fine if GPU memory is tight. Run it with `vLLM` for speed.
- In the prompt, ask for questions an analyst would ask, without reusing the heading's words. Otherwise the task turns into keyword matching.
- Label each question with its paragraph's section id.

**1.8 Filter the questions.**
- Remove exact duplicates and near-duplicates (embedding cosine similarity above about 0.9).
- Have the LLM check each question against its paragraph ("Can this question be answered from this paragraph?") and drop the ones it says no to.
- Read 100 random questions yourself and count how many are good. If fewer than about 80% are, fix the prompt and regenerate.

**1.9 Write about 100 questions yourself.**
- Read the 8 training companies' FY2025 filings as an analyst would, and write questions you would actually ask. Label each with its correct section.
- Spread them across Items 1, 1A, 7 and 8.
- These are your most realistic test, because synthetic questions tend to be easier.

**1.10 Split into train, dev and test.**
- **Held-out companies:** all their questions go into `test_heldout.jsonl`. None are used for training.
- **The other 8 companies:** split 80 / 10 / 10 into train, dev and test, by paragraph rather than by question. Questions written from the same paragraph then never land on both sides of the split.
- **Your hand-written questions:** these go into `test_handwritten.jsonl` and are never used for training.
- Do this separately for FY2025 and FY2024, because objective 4 trains on FY2024.

### Result when complete

- **Parsed filings:** 20 checked filings in `data/parsed/` (10 companies × 2 years), each with `toc.json` and `content.jsonl`.
- **Question files, for each year:**
  - `train`
  - `dev`
  - `test_synthetic`
  - `test_handwritten` (about 100 questions)
  - `test_heldout`
- **Parser:** `src/parse_10k.py`, which turns any 10-K into those two files with one command.
- **Dataset table** for the write-up:

| Company | Year | Leaf sections | ToC length (tokens) | Train | Dev | Test | Parser fixes needed |
|---|---|---|---|---|---|---|---|

---

## Objective 2: Train STAIR on the filings (2–3 days, much of it GPU time)

**What it answers:** can STAIR's recipe be reproduced on 10-K filings?

### Steps

**2.1 Build the training prompts.**
- Follow the paper's Table 2. Each prompt has:
  - an instruction that names the company
  - the filing's full table of contents, one numbered title per line, indented by level
  - the question
- The target is the correct leaf section exactly as written in the ToC, e.g. "7.3 Liquidity and Capital Resources".
- Each question becomes one training example, paired with its own company's ToC.

**2.2 Set up QLoRA.**
- Load Mistral-7B-Instruct-v0.2 in 4-bit (`bitsandbytes`, NF4).
- Add LoRA adapters with the paper's settings (rank 16, alpha 32) on the attention layers.
- Train only on the answer tokens, not the prompt. TRL's `SFTTrainer` does this by default when the data is in prompt–completion form.
- Set the maximum length to your longest prompt plus a margin.

**2.3 Time one epoch first, then set the schedule.**
- The paper allowed up to 200 epochs with early stopping. That won't fit your time.
- Time one epoch, then cap training at whatever fits overnight (likely 5–15 epochs).
- Save a checkpoint every epoch, and stop when dev R@1 hasn't improved for 3 epochs.

**2.4 Restrict the output to real sections.**
- For each filing, build a trie of its leaf titles' token sequences. Use exactly the token sequences of the training targets, each followed by end-of-sequence.
- Pass the trie to Hugging Face `generate` through `prefix_allowed_tokens_fn`. At each step the model may then only produce a token that continues a real title.
- For a top 3, use beam search with 3–5 beams and keep the top 3 distinct sections.
- vLLM's guided decoding with a fixed list of choices does the same job faster.

**2.5 Pick the best checkpoint.**
- Evaluate every saved checkpoint on the dev set with restricted output. A 300-question sample is enough.
- Keep the checkpoint with the highest R@1.

**2.6 Measure invalid outputs.**
- Run the best checkpoint once on the test set without the restriction, and count how often it outputs a title that isn't in the ToC.
- The paper reported 0.05% for STAIR. With the restriction on, the rate is zero by design, so the unrestricted number is the one worth reporting.

### Result when complete

- **Model:** the best LoRA adapter, in `checkpoints/stair_fy2025/`.
- **Figure:** dev R@1 over epochs, for the write-up.
- **Retrieval script:** `src/stair_retrieve.py`. Give it a company and a question, and it returns the top 3 sections.
- **Recorded numbers:**
  - dev R@1
  - unrestricted invalid-output rate
  - training time
  - GPU memory used

---

## Objective 3: Compare STAIR with the alternatives (3 days)

**What it answers:** the main research question. Does STAIR find the right 10-K section more often than standard RAG retrieval, and at what cost?

### Steps

**3.1 Write the scoring script first.**
- Every method produces the same output: for each question, a ranked list of section ids.
- The script compares that list with the correct section and reports R@1, R@3 and nDCG@3, using `ranx`.
- With one correct section per question, R@1 is simply the share of questions where the top answer is right.
- Test the script on a made-up run whose scores you already know.

**3.2 BM25 (keyword RAG).**
- Index each leaf section's text with `rank_bm25`, with a separate index per filing.
- Search only the filing of the company the question is about.

**3.3 Embedding retrieval (dense RAG).**
- Cut long sections into chunks of about 300 words and embed them. NV-Embed-v2 matches the paper; `bge-large-en-v1.5` is a lighter option.
- Score each section by its best-matching chunk.

**3.4 LLM given the ToC, without fine-tuning.**
- Send the STAIR prompt to the untuned Mistral. The paper got 13.8% R@1 this way.
- Send it to one stronger model you have access to as well. The paper used Claude Haiku 4.5.
- Count invalid outputs for both.

**3.5 LLM given the whole 10-K.**
- Send the full filing text, with each section marked by its id and title, and ask which section answers the question.
- A 10-K is often 50,000–100,000 tokens. This needs a long-context model and is the slowest and most expensive method.
- Run it only on the hand-written questions plus a random sample of about 300 synthetic ones.

**3.6 DSI.**
- Train the same model with the same settings, except that the input is only "company + question" (no ToC) and the output is the section id.
- DSI has to memorise which section ids exist. It can't answer anything about the held-out companies, which is worth stating in the write-up.
- Count its invalid outputs without restriction, as for STAIR. The paper reported 3.25%.

**3.7 Significance test.**
- For STAIR against each other method, run a paired randomization test on per-question R@1 (Smucker et al., 2007).
- `ranx` has it built in as Fisher's randomization test.
- Report the p-values.

**3.8 Cost and speed.** For each method, record seconds per question and cost per 1,000 questions (GPU hours or API charges).

**3.9 Break down the results**
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
| DSI | | | n/a | | | | | | |
| **STAIR** | | | | | | | | | – |

It also gives a one-sentence answer to the main question, for example: "On 10-K filings, STAIR's R@1 is X points above the best RAG retriever (p = …)." If STAIR doesn't win, that is still a valid result, and the breakdown in 3.9 explains why.

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

## Objective 6 (stretch): Scale to 25–30 companies (3–4 days)

**What it answers:** when the number of companies triples, does the approach still work, and does the parser still work without hand-fixing?

### Steps

**6.1** Add 15–20 companies from the same sector, again with December fiscal years.

**6.2** Run the parser with no hand-fixing. Check a random 20% of the new ToCs by hand and record the error rate.

**6.3** Generate questions for the new companies, and retrain STAIR and DSI on the larger set.

**6.4** Rerun the comparison (objective 3) and the new-filings test (objective 4).

### Result when complete

- **Results table:** a new group of rows for 25–30 companies, next to the 10-company results.
- **Parser:** its error rate without hand-fixing.
- **Training:** how much training time grew.
- **Conclusion:** one sentence on whether accuracy held. STAIR only ever sees one company's ToC at a time, so it should hold up. DSI has to memorise three times as many section ids, so it may drop.

---

## Time

| Part | Working days (8 h) |
|---|---|
| Before you start | 1 |
| Objective 1 | 4–5 |
| Objective 2 | 2–3 |
| Objective 3 | 3 |
| Objective 4 | 1.5–2 |
| Objective 5 | 2 |
| Objective 6 (stretch) | 3–4 |
| Write-up | 3 |
| **Total** | **19.5–23** |

At 40 hours a week, that is about 4–4.5 weeks.

If time runs short, cut in this order:
1. Objective 6.
2. DSI in step 3.6, which saves about a day.
3. Objective 4.

Keep objectives 1, 2, 3 and 5 and the write-up. They answer both "does it work?" and "what is it for?"
