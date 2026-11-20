# STAIR for 10-K filings: plan

At about 40 hours a week.
Based on: STAIR, arXiv 2609.03874 (`papers/Alternate STAR.pdf`, kept out of git).
- The authors' code and data are public as of the paper's v2: [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean) and Hugging Face `meghanadh/SearchTome`.
- The project builds on that code and adapts it for 10-Ks. CLAUDE.md lists the changes it needs.

## How the work is laid out

1. **Stage 1, the core:** a complete working system on 10 companies. This is the safety net: whatever happens later, you have this.
2. **Stage 2, add-ons:** added one at a time, in order. Finish one before starting the next.
3. **Stage 3, analysis and write-up.**

**Out of scope:** a model that updates itself as new filings arrive. That's an unsolved research problem (see DSI++). Add-on A tests the cheap version of it instead.

**Not in scope:** training DSI or any other baseline. The paper's STAIR vs. DSI result is cited, not reproduced.

---

# Stage 1: The core

## Phase 0: Reading + setup
- [ ] Read the 8 core papers (see the reading list at the bottom)
- [x] Choose a sector and 10 companies: Technology; 8 training companies and 2 held-out first-time filers (CRWV, FIG), listed in `config.yaml`
- [x] Get the GPU details: NVIDIA RTX 4000 SFF Ada, 20 GB, Windows school computer
- [ ] On the GPU machine (Windows, Python 3.15.2, no admin rights): install VS Code (user installer) and `uv`, then PyTorch's CUDA 13.0 build (`--index-url https://download.pytorch.org/whl/cu130`) and `requirements.txt`; check `torch.cuda.is_available()` is `True`
- [x] Give this project its own git repo: [github.com/munashemacdonald/STAIR](https://github.com/munashemacdonald/STAIR)
- [ ] Fork the official STAIR code and clone it on the GPU machine. In the fork:
  - point `uv` at PyTorch's CUDA 13.0 build
  - add `bitsandbytes`
  - start training with `python` instead of `torchrun` on one GPU

  Then run its bundled example end to end.

**Done when:** the official code's bundled example trains and evaluates on the GPU.

## Phase 1: Filings → table of contents + section text
- [ ] Download FY2025 **and** FY2024 10-Ks for the 10 companies from EDGAR (`edgartools` or `sec-edgar-downloader`; send a User-Agent with your contact). FY2024 is for add-on A.
- [ ] Split each FY2025 filing into Items
- [ ] Pull out the headings inside Items 1, 1A, 7, 7A and 8 (including the notes to the financial statements)
- [ ] Save per filing: `toc.json` in the official code's format (`title` plus `table_of_contents` entries with `id`, `section_num`, `title`) and `content.jsonl` (section → text)
- [ ] Check and fix each table of contents by hand
- [ ] Write the parser as reusable code, not a notebook: add-ons A and B run it again

**Done when:** 10 clean tables of contents, and every section has text.
**Checkpoint:** if headings inside Items are still a mess, stop at Items plus numbered Notes and move on.

## Phase 2: Questions
- [ ] Write 3–5 questions per paragraph, labelled with the section each came from: use the official code's question generator through the Hugging Face API, or `transformers` in 4-bit on the GPU
- [ ] Remove duplicates and questions that several sections could answer
- [ ] Split into train / dev / test
- [ ] Hand-written test set: ask FinanceBench's 25 analyst questions (`data/questions/financebench_templates.jsonl`) of every company's FY2025 filing, held-out companies included, and label every section that answers each one
- [ ] Set 2 companies completely aside, to test on filings the model has never seen

**Done when:** train, dev and test files exist, plus the set-aside companies.

## Phase 3: Comparisons + scoring script
None of these comparisons is trained. They're the yardstick that shows whether fine-tuning STAIR helped.

- [ ] Scoring script: R@1, R@3, nDCG@3 (`ranx` or `pytrec_eval`)
- [ ] BM25 keyword search over the sections
- [ ] Embedding-based retriever
- [ ] LLM given the table of contents, picking a section without fine-tuning
- [ ] LLM given the whole 10-K

**Done when:** the first results table exists.
**Checkpoint:** if the whole-10-K approach is already very accurate, STAIR's case rests on cost, speed and never pointing to a section that doesn't exist. Measure those too.

## Phase 4: Train STAIR
- [ ] Add 4-bit (QLoRA) loading to the official code, so Mistral-7B fits in 20 GB
- [ ] Reproduce one SearchTome domain (finance) with the official code and compare with the paper's R@1
- [ ] Adapt the official code for 10-Ks: each question paired with its own filing's ToC, a prompt that names the company and filing, and a longer maximum length
- [ ] QLoRA fine-tune, keeping the best checkpoint (dev R@1 if possible; the code's default is dev loss)
- [ ] Restricted output, as our extension: a trie of the real section titles (`prefix_allowed_tokens_fn`). Also report the official unrestricted evaluation.
- [ ] Score on three test sets: LLM-written questions, your own questions, the set-aside companies

**Done when:** STAIR is a row in the results table.

## Phase 5: Answers, command-line demo
- [ ] Send the section STAIR picks to a second LLM that answers and cites the section
- [ ] Run it from the command line

**Done when:** you type a question and get an answer with the section it came from.

### Milestone
Working system + results table. If the core isn't finished, finish it before starting any add-on.

---

# Stage 2: Add-ons, in order

Start the long GPU runs first and do the data work while they train.

**Buffer:** if the core runs late, use this time to finish it. Otherwise it's room for the investment demos: an analyst checklist across companies, and a side-by-side view of what changed between each company's FY2024 and FY2025 filings.

## A. Year-over-year test
- [ ] Run the parser on the FY2024 10-Ks, then generate questions for them
- [ ] Train on FY2024, test on FY2025 questions (FY2025 table of contents in the prompt) without retraining

**Done when:** you can say how much accuracy drops on next year's filing. This is the cheap answer to "does it hold up on new filings?"

## B. Scale to 25–30 companies
- [ ] Make the parser work without hand-fixing; spot-check a random sample instead
- [ ] Generate questions, retrain, and rescore

**Done when:** the results table has a 25–30-company row.

## Stretch goals (only if you're ahead of schedule)
- [ ] **C. XBRL figures in answers:** pull reported numbers from the XBRL company-facts API instead of the text
- [ ] **D. 10-Qs:** these have a different structure, and there are three per year per company
- [ ] **E. Polished interface:** Streamlit or Gradio

---

# Stage 3: Analysis + write-up
- [ ] Mistakes: wrong section vs. a section that doesn't exist; which kinds of questions fail
- [ ] Write up results, limitations, next steps (self-updating model, stretch goals not reached)
- [ ] In the methods section, describe the hand-written test set as FinanceBench's 25 domain-relevant questions (Islam et al., 2023), adapted to each company's FY2025 10-K, with your section labels; say how many were skipped and why

---

## Risks and what to do about each

| Risk | What to do |
|---|---|
| Headings inside the Items are too messy to extract | Stop at Items plus numbered Notes, and fix the rest by hand |
| GPU access is delayed | Train a smaller 1–3B model first; Phases 1 and 3 run on the laptop |
| The official code won't run on Windows | The setup already covers the known problems: CUDA build, no `torchrun`, UTF-8. If something else fails, check `torch.cuda.is_available()` first, then read the error before changing code |
| The authors don't grant a licence | Keep their code in a separate fork; publish only your own code and a list of changes |
| Training runs take longer than planned | Fewer epochs, shorter tables of contents, and run training overnight |
| The fine-tuned model doesn't beat the simpler approaches | Still a valid result: report it honestly and explain why |
| Behind schedule | Drop add-ons from the bottom up; never cut the core or the write-up |

## Reading

**The 8 core papers:**
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
