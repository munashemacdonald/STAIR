# STAIR for 10-K filings: plan

At about 40 hours a week.
Based on: STAIR, arXiv 2609.03874 (`papers/Alternate STAR.pdf`, kept out of git).
- The authors' code and data are public as of the paper's v2: [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean) and Hugging Face `meghanadh/SearchTome`.
- The project builds on that code and adapts it for 10-Ks. CLAUDE.md lists the changes it needs.

This checklist follows the same six objectives as `OBJECTIVES_PLAN.md` and the one-pager. The numbers in brackets, like (1.8), are the steps in the objectives plan, where each item is explained.

## Order of work

1. **The core, never cut:** setup, then objectives 1, 2, 3 and 5, then the write-up.
2. **If time allows:** objective 4, then objective 6. If time runs short, cut 6 first, then 4.

**Out of scope:**
- a model that updates itself as new filings arrive (an unsolved research problem; see DSI++). Objective 4 tests the cheap version instead.
- training DSI or any other baseline. The paper's STAIR vs. DSI result is cited, not reproduced.
- the analyst agent itself, and predicting stock prices or testing trading returns

---

## Setup
- [ ] Read the 8 core papers (see the reading list at the bottom)
- [x] Choose a sector and companies: Technology; 8 training companies, 2 held-out first-time filers (CRWV, FIG) and 2 established test-only companies (ORCL, AMD), listed in `config.yaml`
- [x] Get the GPU details: NVIDIA RTX 4000 SFF Ada, 20 GB, Windows school computer
- [ ] On the GPU machine (Windows, Python 3.15.2, no admin rights): install VS Code (user installer) and `uv`, then PyTorch's CUDA 13.0 build (`--index-url https://download.pytorch.org/whl/cu130`) and `requirements.txt`. Check that `torch.cuda.is_available()` is `True`.
- [x] Give this project its own git repo: [github.com/munashemacdonald/STAIR](https://github.com/munashemacdonald/STAIR)
- [ ] Fork the official STAIR code and clone it on the GPU machine. In the fork:
  - point `uv` at PyTorch's CUDA 13.0 build
  - add `bitsandbytes`
  - start training with `python` instead of `torchrun` on one GPU

  Then run its bundled example end to end.

**Done when:** the official code's bundled example trains and evaluates on the GPU.

---

## Objective 1: Build a 10-K question dataset

**Filings → tables of contents and section text**
- [ ] Download 20 10-Ks from EDGAR with `edgartools`, sending a User-Agent with your contact (1.1):
  - FY2024 and FY2025 for the 8 training companies
  - FY2025 only for the 2 held-out and 2 test-only companies
- [ ] Split each filing into Items (1.2)
- [ ] Pull out the headings inside Items 1, 1A, 7, 7A and 8, including the notes to the financial statements (1.3)
- [ ] Save per filing: `toc.json` in the official code's format (`title` plus `table_of_contents` entries with `id`, `section_num`, `title`), and `content.jsonl` (section → text) (1.4–1.5)
- [ ] Check and fix each table of contents by hand, and record how many fixes each needed (1.6)
- [ ] Write the parser as reusable code, not a notebook: objectives 4 and 6 run it again

**Checkpoint:** if headings inside Items are still a mess, stop at Items plus numbered Notes and move on.

**Generated questions**
- [ ] Write 3 questions per paragraph, labelled with the section each came from (1.7). Use the official code's question generator through the Hugging Face API, or `transformers` in 4-bit on the GPU, with a model whose licence lets its outputs train STAIR (Qwen2.5-14B-Instruct is Apache 2.0).
- [ ] Keep the hand-written test questions out of generation: never use them or their templates as prompt examples, and drop generated questions that closely match them (cosine above about 0.85) (1.7)
- [ ] Remove duplicates. Then, for each dev and test question, have an LLM judge check the top 3 other sections from BM25 and embeddings, and add any that also answer it as extra correct sections. Check 10% of training questions the same way. (1.8)

**Hand-written test set: the headline test** (1.9)
- [ ] Ask these of every company's FY2025 filing, held-out and test-only companies included: 12 companies, up to about 450 questions:
  - FinanceBench's 25 analyst questions (`data/questions/financebench_templates.jsonl`)
  - the 8 checklist questions (`data/questions/checklist_templates.jsonl`)
  - 5 company-specific questions per company, written before opening that company's table of contents
- [ ] Second labeller: a classmate labels about 30 random questions with the rule in `data/questions/README.md`, without seeing your labels. Compare, settle disagreements, tighten the rule, then label the rest ("answers on its own" vs "needed together").

**Splits** (1.10)
- [ ] Split the 8 training companies' questions into train / dev / test by paragraph, separately for FY2024 and FY2025
- [ ] Put all held-out and test-only questions in `test_heldout.jsonl`, with a `group` field: `first_time` (CRWV, FIG) or `established` (ORCL, AMD)

**Done when:** 20 clean tables of contents; train, dev and test files; the hand-written set labelled, with the second-labeller agreement.

---

## Objective 2: Train STAIR on the filings
- [ ] Add 4-bit (QLoRA) loading to the official code, so Mistral-7B fits in 20 GB (2.2)
- [ ] Reproduce one SearchTome domain (finance) with the official code, and compare with the paper's R@1 (2.3)
- [ ] Adapt the official code for 10-Ks: each question paired with its own filing's ToC, a prompt that names the company and filing, and a longer maximum length (2.4)
- [ ] Add a `--resume` option to the fork's `train` command, and test it once: stop a run after its first checkpoint, then resume it (2.5)
- [ ] Check the training time against the budget, at most about 8 hours per run, estimated from a 200-step timing. If it's over, cut in this order: cap the questions per section, then a two-level ToC, then fewer epochs, then a smaller model. (2.5)
- [ ] QLoRA fine-tune, saving a checkpoint about every 30 minutes to a drive that persists between logins, and keeping the best checkpoint (dev R@1 if possible; the code's default is dev loss) (2.5–2.6)
- [ ] Restricted output, as our extension: a trie (`prefix_allowed_tokens_fn`) over the full list strings the model was trained on, `["<section_num> <title>"]`, tokenized exactly like the training targets. Its top 3 is the 3 best of 5 beams. Also report the official unrestricted evaluation, whose top 3 is the first 3 valid items of its generated list. (2.7)

**Done when:** the best checkpoint is chosen, the SearchTome reproduction is recorded, and invalid outputs are reported for both versions.

---

## Objective 3: Compare STAIR with the alternatives
None of these comparisons is trained. They're the yardstick that shows whether fine-tuning STAIR helped.

- [ ] Scoring script: R@1 and R@3 (any section that answers on its own is in the top k), nDCG@3, and All@3 for needed-together questions (3.1)
- [ ] BM25 keyword search over the sections (3.2)
- [ ] Embedding-based retriever (3.3)
- [ ] Hybrid: BM25 + embeddings, merged with reciprocal rank fusion (k = 60) (3.3a)
- [ ] Hybrid + reranker: `BAAI/bge-reranker-v2-m3` over the hybrid's top 20 sections. Also report the hybrid's R@20, the most the reranker can reach. (3.3b)
- [ ] LLM given the table of contents, without fine-tuning: the untuned Mistral and one stronger API model. Record the API model's training-data cutoff. (3.4)
- [ ] LLM given the whole 10-K, on the hand-written questions plus a 300-question synthetic sample. Score every other method on the same 300 too, and record this model's cutoff. (3.5)
- [ ] Pooling: once every method has run, judge each method's top 3 sections that aren't labelled yet, without knowing which method found them. Record the yeses in `pooled_answers_alone`, then rescore every method, with and without them. (3.6)
- [ ] Statistics (3.7):
  - **Primary comparison,** fixed in `config.yaml` before any test: STAIR (official) against hybrid + reranker, R@1 on the hand-written set, with the confidence interval of the difference. Resample templates and company-specific questions as units.
  - **Secondary comparisons:** Holm-adjusted p-values.
  - **Other test sets:** resample sections.
  - **Small cells:** under about 30 questions, show the count and confidence interval, and draw no conclusion from them.
  - **Freeze the test sets:** make every choice on dev.
- [ ] Cost and speed for each method (3.8)
- [ ] Breakdowns (3.9):
  - by test set, with the held-out groups split into first-time filers and established companies
  - by Item, and by the number of correct sections
  - by template group
  - for the API models, by filing date relative to their training cutoffs

**Checkpoint:** if the whole-10-K approach is already very accurate, STAIR's case rests on cost, speed and never pointing to a section that doesn't exist. Measure those too.

**Done when:** the main results table and the like-for-like table exist, after pooling, with the primary comparison's confidence interval.

---

## Objective 4: Test whether it holds up on new filings (if time allows)
- [ ] Train STAIR on the FY2024 questions, and test on FY2025 questions with the FY2025 table of contents, without retraining. Run both versions, and count FY2024 titles among the invalid outputs. (4.1–4.2)
- [ ] Label each FY2025 section as unchanged, renamed or new, and break down accuracy by that label, showing the counts (4.3–4.4)
- [ ] Report the held-out first-time filers and the established test-only companies here as well (4.5)

**Done when:** you can say how much accuracy drops on next year's filing. This is the cheap answer to "does it hold up on new filings?"

---

## Objective 5: Show the investment use
- [ ] Write a 1–5 scale in plain words for each checklist question (5.1)
- [ ] Run the checklist for every company and year with STAIR's top 3 sections, then again with hybrid + reranker's top 3, using the same answering model and prompt (5.2)
- [ ] Add the year-over-year change (5.3)
- [ ] Check the quality (5.4):
  - repeatability, by rewording each question
  - your own scores on about 20 pairs
  - the right section
  - the spread of scores
  - the WRDS checks where your subscription has the data: Audit Analytics for material weaknesses, and Compustat for customers, debt maturities and buybacks
  - STAIR against hybrid + reranker
- [ ] Write 2–3 case studies (5.5)
- [ ] Cited-answer check: about 50 hand-written questions answered from each pipeline's top 3 sections, judged for correct, supported and a valid citation. Check 20 of the judge's decisions yourself. (5.6)
- [ ] Command-line demo: type a question, and get an answer with the section it came from

**Done when:** the signal table, the pipeline comparison and the cited-answer check exist.

---

## Objective 6 (stretch): Scale to 25–30 companies
- [ ] Make the parser work without hand-fixing; spot-check a random sample instead
- [ ] Generate questions, retrain, and rescore

**Done when:** the results table has a 25–30-company row.

## Beyond the objectives (only if you're ahead of schedule)
- [ ] **XBRL figures in answers:** pull reported numbers from the XBRL company-facts API instead of the text
- [ ] **10-Qs:** these have a different structure, and there are three per year per company
- [ ] **Polished interface:** Streamlit or Gradio

---

## Write-up
- [ ] Mistakes: wrong section vs. a section that doesn't exist; which kinds of questions fail
- [ ] Write up results, limitations and next steps. Limitations to cover:
  - STAIR is the only method trained on these filings
  - API models may have seen some filings during training
  - the project is non-commercial (FinanceBench and NV-Embed-v2 licences)
  - the held-out groups are small
- [ ] In the methods section, describe the hand-written test set as FinanceBench's 25 domain-relevant questions (Islam et al., 2023), adapted to each company's FY2025 10-K, plus the 8 checklist and 5 company-specific questions per company, with your section labels. Say how many were skipped and why, report the second-labeller agreement, and say how many sections pooling added for each method.
- [ ] List what each part is licensed under (objectives plan, "Licences")

---

## Risks and what to do about each

| Risk | What to do |
|---|---|
| Headings inside the Items are too messy to extract | Stop at Items plus numbered Notes, and fix the rest by hand |
| GPU access is delayed | Train a smaller 1–3B model first; objectives 1 and 3 run on the laptop |
| The official code won't run on Windows | The setup already covers the known problems: CUDA build, no `torchrun`, UTF-8. If something else fails, check `torch.cuda.is_available()` first, then read the error before changing code |
| The authors don't grant a licence | Keep their code in a separate fork; publish only your own code and a list of changes |
| Training runs take longer than planned | Check the budget before the full run (at most about 8 hours). If it's over, cut in this order: questions per section, a two-level ToC, epochs, then model size |
| The school computer stops a run (logoff, restart, update) | Checkpoints about every 30 minutes on a drive that persists; resume from the latest checkpoint instead of restarting |
| WRDS doesn't have a source the checklist needs | Use your own scores for that question, and say so |
| The fine-tuned model doesn't beat the simpler approaches | Still a valid result: report it honestly and explain why |
| Behind schedule | Cut objective 6, then objective 4; never cut the core or the write-up |

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

**When you reach that point:**

| When | Read |
|---|---|
| Objective 1 | SEC "How to Read a 10-K" (Item structure) |
| Objective 3 | Reciprocal rank fusion (Cormack et al., 2009); randomization tests (Smucker et al., 2007) |
| Write-up | RAG (Lewis et al., 2020); "Lost in the Middle" (Liu et al., 2024); RAPTOR (Sarthi et al., 2024) |
