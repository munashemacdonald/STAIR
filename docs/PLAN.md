# STAIR for 10-K filings: one-month plan

Timeline: Nov 2 – Nov 30, 2026, at about 40 hours a week.
Based on: STAIR, arXiv 2609.03874 (`papers/Alternate STAR.pdf`, kept out of git). The paper's code and data link has expired, so everything here is rebuilt from the paper.

## How the month is laid out

1. **Stage 1, the core (Nov 2–15):** a complete working system on 10 companies. This is the safety net: whatever happens later, you have this.
2. **Stage 2, add-ons (Nov 16–25):** added one at a time, in order. Finish one before starting the next.
3. **Stage 3, analysis and write-up (Nov 27–30).** Thanksgiving is Nov 26.

**Not this month:** a model that updates itself as new filings arrive. That's an unsolved research problem (see DSI++). Add-on A tests the cheap version of it instead.

**Not in scope:** training DSI or any other baseline (decided Nov 9). The paper's STAIR vs. DSI result is cited, not reproduced.

---

# Stage 1: The core (Nov 2–15)

## Phase 0: Reading + setup (Nov 2–4)
- [ ] Nov 2–3: read the 8 core papers (see the reading list at the bottom)
- [ ] Choose a sector and 10 companies (avoid banks, whose 10-Ks are very long)
- [x] Get the GPU details: NVIDIA RTX 4000 SFF Ada, 20 GB, Windows school computer (Nov 17)
- [ ] On the GPU machine (Windows, no admin rights): install VS Code (user installer) and Miniconda, then PyTorch with CUDA and `pip install -r requirements.txt`; check `torch.cuda.is_available()`
- [x] Give this project its own git repo: [github.com/munashemacdonald/STAIR](https://github.com/munashemacdonald/STAIR)

**Done when:** a tiny practice fine-tune runs on the GPU.

## Phase 1: Filings → table of contents + section text (Nov 5–7)
- [ ] Download FY2025 **and** FY2024 10-Ks for the 10 companies from EDGAR (`edgartools` or `sec-edgar-downloader`; send a User-Agent with your contact). FY2024 is for add-on A.
- [ ] Split each FY2025 filing into Items
- [ ] Pull out the headings inside Items 1, 1A, 7, 7A and 8 (including the notes to the financial statements)
- [ ] Save per filing: `toc.json` (section ID, title, position in the outline) and `content.jsonl` (section → text)
- [ ] Check and fix each table of contents by hand
- [ ] Write the parser as reusable code, not a notebook: add-ons A and B run it again

**Done when:** 10 clean tables of contents, and every section has text.
**Checkpoint (Nov 7):** if headings inside Items are still a mess, stop at Items plus numbered Notes and move on.

## Phase 2: Questions (Nov 8–9)
- [ ] Use an open LLM on the GPU (`transformers`, 4-bit, batched) to write 3–5 questions per paragraph, labelled with the section each came from
- [ ] Remove duplicates and questions that several sections could answer
- [ ] Split into train / dev / test
- [ ] Write about 100 questions yourself, the way an analyst would ask them, each labelled with the correct section
- [ ] Set 2 companies completely aside, to test on filings the model has never seen

**Done when:** train, dev and test files exist, plus the set-aside companies.

## Phase 3: Comparisons + scoring script (Nov 10–11)
None of these comparisons is trained. They're the yardstick that shows whether fine-tuning STAIR helped.

- [ ] Scoring script: R@1, R@3, nDCG@3 (`ranx` or `pytrec_eval`)
- [ ] BM25 keyword search over the sections
- [ ] Embedding-based retriever
- [ ] LLM given the table of contents, picking a section without fine-tuning
- [ ] LLM given the whole 10-K

**Done when:** the first results table exists.
**Checkpoint (Nov 11):** if the whole-10-K approach is already very accurate, STAIR's case rests on cost, speed and never pointing to a section that doesn't exist. Measure those too.

## Phase 4: Train STAIR (Nov 12–14)
- [ ] Format prompts like the paper's Table 2 (table of contents + question in, section title out)
- [ ] QLoRA fine-tune, keeping the checkpoint with the best dev R@1
- [ ] Restricted output: a trie of the real section titles (`prefix_allowed_tokens_fn`), so the model can only produce a heading that exists
- [ ] Score on three test sets: LLM-written questions, your own questions, the set-aside companies

**Done when:** STAIR is a row in the results table.

## Phase 5: Answers, command-line demo (Nov 15)
- [ ] Send the section STAIR picks to a second LLM that answers and cites the section
- [ ] Run it from the command line

**Done when:** you type a question and get an answer with the section it came from.

### Milestone (Nov 15)
Working system + results table. If the core isn't finished by now, spend Stage 2 finishing it before starting any add-on.

---

# Stage 2: Add-ons, in order (Nov 16–25)

Start the long GPU runs first and do the data work while they train.

**Nov 16–18 are open** (the DSI add-on was dropped on Nov 9). Use them as a buffer if the core runs late, or for the investment demos: an analyst checklist across companies, and a side-by-side view of what changed between each company's FY2024 and FY2025 filings. Otherwise, start add-on A early.

## A. Year-over-year test (Nov 19–20)
- [ ] Run the parser on the FY2024 10-Ks, then generate questions for them
- [ ] Train on FY2024, test on FY2025 questions (FY2025 table of contents in the prompt) without retraining

**Done when:** you can say how much accuracy drops on next year's filing. This is the cheap answer to "does it hold up on new filings?"

## B. Scale to 25–30 companies (Nov 21–25)
- [ ] Make the parser work without hand-fixing; spot-check a random sample instead
- [ ] Generate questions, retrain, and rescore

**Done when:** the results table has a 25–30-company row.

## Stretch goals (only if you're ahead of schedule)
- [ ] **C. XBRL figures in answers:** pull reported numbers from the XBRL company-facts API instead of the text
- [ ] **D. 10-Qs:** these have a different structure, and there are three per year per company
- [ ] **E. Polished interface:** Streamlit or Gradio

---

# Stage 3: Analysis + write-up (Nov 27–30)
- [ ] Mistakes: wrong section vs. a section that doesn't exist; which kinds of questions fail
- [ ] Write up results, limitations, next steps (self-updating model, stretch goals not reached)

---

## Risks and what to do about each

| Risk | What to do |
|---|---|
| Headings inside the Items are too messy to extract | Stop at Items plus numbered Notes, and fix the rest by hand |
| GPU access is delayed | Train a smaller 1–3B model first; Phases 1 and 3 run on the laptop |
| Training runs take longer than planned | Fewer epochs, shorter tables of contents, and run training overnight |
| The fine-tuned model doesn't beat the simpler approaches | Still a valid result: report it honestly and explain why |
| Behind schedule | Drop add-ons from the bottom up; never cut the core or the write-up |

## Reading

**The 8 core papers (Nov 2–3):**
1. DSI: "Transformer Memory as a Differentiable Search Index" (Tay et al., 2022)
2. DSI++: "Updating Transformer Memory with New Documents" (Mehta et al., 2022)
3. LoRA (Hu et al., 2021)
4. QLoRA (Dettmers et al., 2023)
5. GENRE: "Autoregressive Entity Retrieval" (De Cao et al., 2021), for restricted output
6. DPR: "Dense Passage Retrieval" (Karpukhin et al., 2020)
7. InPars (Bonifacio et al., 2022), for generating questions
8. FinanceBench (Islam et al., 2023)

**When you reach that phase:**

| When | Read |
|---|---|
| Phase 1 | SEC "How to Read a 10-K" (Item structure) |
| Stage 3 | RAG (Lewis et al., 2020); "Lost in the Middle" (Liu et al., 2024); RAPTOR (Sarthi et al., 2024) |
