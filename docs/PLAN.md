# STAIR for 10-K filings: plan

At about 40 hours a week.
Based on: STAIR, arXiv 2609.03874 (`papers/Alternate STAR.pdf`, kept out of git).
- The authors' code and data are public as of the paper's v2: [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean) and Hugging Face `meghanadh/SearchTome`.
- The project builds on that code and adapts it for 10-Ks. CLAUDE.md lists the changes it needs.

This checklist follows the same six objectives as `OBJECTIVES_PLAN.md` and the project overview. The numbers in brackets, like (1.8), are the steps in the objectives plan, where each item is explained.

## Order of work

Re-planned on Nov 25 with about 15 days left, after a critical review. The objectives plan lists the changes ("Changes after the critical review").

1. **The core, never cut:** setup, then objectives 1, 2 and 3, objective 4's breakdowns, objective 5, then the write-up. STAIR trains on FY2024 only, so objective 3 already tests it on next year's filings.
2. **If time allows:** the retrained model (4.3), then the SearchTome reproduction (2.3).
3. **After this project:** objective 6.

**Out of scope:**
- a model that updates itself as new filings arrive (an unsolved research problem; see DSI++). Objective 4 tests the cheap version instead.
- training DSI or any other baseline. The paper's STAIR vs. DSI result is cited, not reproduced.
- the analyst agent itself, and predicting stock prices or testing trading returns

---

## Setup
- [ ] Read the 8 core papers (see the reading list at the bottom)
- [x] Choose a sector and companies: Technology; 8 training companies, 2 held-out first-time filers (CRWV, FIG) and 2 established test-only companies (ORCL, AMD), listed in `config.yaml`
- [x] Choose where the GPU work runs (Nov 24): Google Colab, paid through the Google AI Pro plan (about 200 compute units a month). The laptop runs every step that needs no GPU. The school GPU (RTX 4000 SFF Ada, 20 GB, Windows) is the fallback.
- [x] Laptop environment (Nov 24): `.venv` on the python.org Python 3.14.0, with CPU-only torch, `requirements.txt` and `ipykernel`. Smart App Control blocks uv's own Python builds, and `ranx` has no Windows builds for 3.15.
- [x] Give this project its own git repo: [github.com/munashemacdonald/STAIR](https://github.com/munashemacdonald/STAIR)
- [ ] Install Google Drive for desktop, and make `My Drive/stair/` with `parsed/`, `questions/`, `checkpoints/`, `runs/` and `logs/`
- [ ] Check Colab: the compute-unit balance shows about 200, and L4 and A100 are offered under Runtime → Change runtime type
- [ ] Add `HF_TOKEN` and a read-only `GITHUB_TOKEN` to Colab Secrets
- [ ] Make the Colab notebook, `notebooks/colab_gpu.ipynb`: mount Drive, read the tokens, clone the repos, install, and check the GPU. Commit it with outputs cleared.
- [ ] Make a private copy of the official STAIR code. A GitHub fork would be public, and the code has no licence. Add `bitsandbytes`, then run its bundled example end to end on a Colab GPU. No Windows changes are needed on Colab.

**Done when:** the official code's bundled example trains and evaluates on a Colab GPU.

---

## Objective 1: Build a 10-K question dataset

**First, before building the parser**
- [ ] Write the 36 company-specific questions, 3 per company, from news and earnings calls, and save them. Each must be written before you see that company's table of contents, and building the parser shows you them. (1.9)

**Filings → tables of contents and section text**
- [ ] Download 20 10-Ks from EDGAR with `edgartools`, sending a User-Agent with your contact (1.1):
  - FY2024 and FY2025 for the 8 training companies
  - FY2025 only for the 2 held-out and 2 test-only companies
- [ ] Split each filing into Items (1.2)
- [ ] Pull out the headings inside Items 1, 1A, 7, 7A and 8, including the notes to the financial statements (1.3)
- [ ] Save per filing: `toc.json` in the official code's format (`title` plus `table_of_contents` entries with `id`, `section_num`, `title`), and `content.jsonl` (section → text) (1.4–1.5)
- [ ] Check and fix each table of contents by hand, and record how many fixes each needed (1.6)
- [ ] Write the parser as reusable code, not a notebook: objectives 4 and 6 run it again

**Checkpoint:** heading extraction is time-boxed to about 1.5 days. If headings inside Items are still a mess by then, stop at Items plus numbered Notes and move on. (1.3)

**Generated questions**
- [ ] Write questions per section, each labelled with the one section it came from (1.7):
  - FY2024, 8 training companies: 3–6 per section, about 5,000 for training
  - FY2025, all 12 companies: 2 per section, for dev and test

  Run Qwen2.5-14B-Instruct-AWQ with vLLM on a Colab L4 (Apache 2.0, so its outputs can train STAIR), or `transformers` in 4-bit if vLLM won't install. Copy the questions from Drive into `data/questions/` and commit them.
- [ ] Keep the hand-written test questions out of generation: never use them or their templates as prompt examples, and drop generated questions that closely match them (cosine above about 0.85) (1.7)
- [ ] Remove duplicates. Then, for each dev and test question, have an LLM judge check the top 3 other sections from BM25 and embeddings, and add any that also answer it as extra correct sections. Check 10% of training questions the same way; a training question keeps one target section or is dropped. (1.8)

**Hand-written test set: the headline test** (1.9)
- [ ] Ask these of every company's FY2025 filing, held-out and test-only companies included: 12 companies, up to 216 questions, about 200 after skips:
  - 7 FinanceBench templates: dg09, dg11, dg14, dg19, dg20, dg22, dg24 (`data/questions/financebench_templates.jsonl`; chosen Nov 25)
  - the 8 checklist questions (`data/questions/checklist_templates.jsonl`)
  - the 36 company-specific questions
- [ ] Second labeller: a classmate labels about 30 random questions with the rule in `data/questions/README.md`, without seeing your labels. Compare, settle disagreements, tighten the rule, then label the rest ("answers on its own" vs "needed together").

**Splits, by year** (1.10)
- [ ] `train.jsonl`: every FY2024 question of the 8 training companies. STAIR never trains on an FY2025 filing.
- [ ] The 8 training companies' FY2025 questions, split by section: 20% to `dev.jsonl`, 80% to `test_synthetic.jsonl`
- [ ] All held-out and test-only questions in `test_heldout.jsonl`
- [ ] Every FY2025 question has a `group` field: `seen_company`, `first_time` (CRWV, FIG) or `established` (ORCL, AMD)

**Done when:** 20 clean tables of contents; train, dev and test files; the hand-written set labelled, with the second-labeller agreement.

---

## Objective 2: Train STAIR on the filings
- [ ] Add 4-bit (QLoRA) loading to the official code, so Mistral-7B fits on an L4 (24 GB). Use it on every GPU, so runs match. (2.2)
- [ ] If time allows: reproduce one SearchTome domain (finance) with the official code, and compare with the paper's R@1 (2.3)
- [ ] Adapt the official code for 10-Ks: each question paired with its own filing's ToC, a prompt that names the company and filing, and a longer maximum length (2.4)
- [ ] Add a `--resume` option to your copy's `train` command, and test it once: stop a run after its first checkpoint, then resume it (2.5)
- [ ] Check the training time against the budget, at most about 8 hours per run, estimated from a 200-step timing. Questions are already capped per section. If it's still over, cut in this order: a two-level ToC, then fewer epochs, then a smaller model. (2.5)
- [ ] Unit-test the trie before training, with the tokenizer only: each target's training tokens must be exactly one path through its filing's trie, and the trie must accept no title outside the filing (2.7)
- [ ] QLoRA fine-tune on the FY2024 training questions, on Colab, with the tab open and the laptop awake. Save a checkpoint about every 30 minutes to Google Drive, and keep the best checkpoint on the FY2025 dev set (dev R@1 if possible; the code's default is dev loss). Record the GPU type and compute units used. (2.5–2.6)
- [ ] Restricted output, as our extension: a trie (`prefix_allowed_tokens_fn`) over the one-item list strings the model was trained on, `["<section_num> <title>"]`, tokenized exactly like the training targets. Its top 3 is the 3 best of 5 beams. Also report the official unrestricted evaluation, whose top 3 is the first 3 valid items of its generated list. (2.7)

**Done when:** the trie test passes, the best checkpoint is chosen, and invalid outputs are reported for both versions.

---

## Objective 3: Compare STAIR with the alternatives
None of these comparisons is trained. They're the yardstick that shows whether fine-tuning STAIR helped. Every method is tested on the FY2025 filings.

- [ ] Scoring script: R@1 and R@3 (any section that answers on its own is in the top k), nDCG@3, and All@3 for needed-together questions (3.1)
- [ ] Chunks for every RAG method: about 300 words, overlapping by 50, each starting with its section number and title; a section scores as its best chunk (MaxP) (3.3)
- [ ] BM25 keyword search: whole sections or chunks, whichever is better on dev (3.2)
- [ ] Embedding-based retriever (3.3)
- [ ] Hybrid: BM25 + embeddings, merged with reciprocal rank fusion (k = 60) (3.3a)
- [ ] Hybrid + reranker: `BAAI/bge-reranker-v2-m3` over the hybrid's top 20 sections. Also report the hybrid's R@20, the most the reranker can reach. (3.3b)
- [ ] LLM given the table of contents, without fine-tuning: the untuned Mistral and one stronger API model. Record the API model's training-data cutoff. The stronger model is the comparison most likely to challenge fine-tuning. (3.4)
- [ ] LLM given the whole 10-K, on the hand-written questions plus a 300-question synthetic sample. Score every other method on the same 300 too, and record this model's cutoff. (3.5)
- [ ] Pooling: once every method has run, the LLM judge checks each method's top 3 sections that aren't labelled yet, without knowing which method found them. For the hand-written set, check 100 of its decisions yourself, blind. Record the yeses in `pooled_answers_alone`, then rescore every method, with and without them. (3.6)
- [ ] Statistics (3.7):
  - **Primary comparison,** fixed in `config.yaml` before any test: STAIR (official) against hybrid + reranker, R@1 on the hand-written set, with the confidence interval of the difference. Resample templates and company-specific questions as units.
  - **Secondary comparisons:** Holm-adjusted p-values.
  - **Other test sets:** resample sections.
  - **Small cells:** under about 30 questions, show the count and confidence interval, and draw no conclusion from them.
  - **Freeze the test sets:** make every choice on dev.
- [ ] Cost and speed for each method, in Colab compute units or API charges, with the GPU type (3.8)
- [ ] Breakdowns (3.9):
  - by test set and group: seen companies in a new year, first-time filers and established companies
  - by Item, and by the number of correct sections
  - by template group
  - for the API models, by filing date relative to their training cutoffs

**Checkpoint:** if the stronger LLM given the ToC, or the whole-10-K method, is already very accurate, STAIR's case rests on running locally (private, no per-question cost, steady speed) and never pointing to a section that doesn't exist. Measure those too.

**Done when:** the main results table and the like-for-like table exist, after pooling, with the primary comparison's confidence interval.

---

## Objective 4: Test whether it holds up on new filings
Objective 3 already tests STAIR, trained on FY2024, on FY2025. This objective breaks the results down.

- [ ] Label each seen company's FY2025 section as unchanged, renamed or new, and break down accuracy by that label, showing the counts. Count FY2024 titles among the official version's invalid outputs. (4.1–4.2)
- [ ] Report the held-out first-time filers and the established test-only companies here as well (4.4)
- [ ] If time allows: train STAIR on the FY2025 filings too, and report the gap to the FY2024-trained model as the cost of not retraining (4.3)

**Done when:** you can say where STAIR's errors on next year's filing come from.

---

## Objective 5: Show the investment use
- [ ] Write a 1–5 scale in plain words for each checklist question (5.1)
- [ ] Run the checklist for every company and year with STAIR's top 3 sections, then again with hybrid + reranker's top 3, using the same answering model and prompt. Cap each section at about 3,000 tokens, keeping a long section's best chunks, with the same rule for both. (5.2)
- [ ] Add the year-over-year change (5.3)
- [ ] Check the quality (5.4):
  - repeatability, by rewording each question
  - your own scores on about 20 pairs
  - the right section
  - the spread of scores
  - the WRDS checks where your subscription has the data: Audit Analytics for material weaknesses, and Compustat for customers, debt maturities and buybacks
  - STAIR against hybrid + reranker
- [ ] Write 2 case studies (5.5)
- [ ] Cited-answer check: about 50 hand-written questions answered from each pipeline's top 3 sections, judged for correct, supported and a valid citation. Check 20 of the judge's decisions yourself. (5.6)
- [ ] Command-line demo: type a question, and get an answer with the section it came from

**Done when:** the signal table, the pipeline comparison and the cited-answer check exist.

---

## Objective 6 (after this project): Scale to 25–30 companies
It doesn't fit in the 15 days left on Nov 25; list it under next steps in the write-up.
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
  - STAIR is the only method trained on these companies' filings (FY2024); the RAG methods index FY2025 directly
  - API models may have seen some filings during training
  - the project is non-commercial (FinanceBench and NV-Embed-v2 licences)
  - the held-out groups are small, and the hand-written set has about 51 independent units
  - STAIR is trained on one section per question, so it isn't trained for needed-together questions
- [ ] In the methods section, describe the hand-written test set as 7 of FinanceBench's 25 domain-relevant questions (Islam et al., 2023), adapted to each company's FY2025 10-K, plus the 8 checklist and 3 company-specific questions per company, with your section labels. Say which 7 and why, how many were skipped and why, report the second-labeller agreement, and say how many sections pooling added for each method.
- [ ] List what each part is licensed under (objectives plan, "Licences")
- [ ] Record where each step ran: the GPU type and compute units for each Colab step, and the Python versions (3.14.0 on the laptop, Colab's own on Colab)

---

## Risks and what to do about each

| Risk | What to do |
|---|---|
| Headings inside the Items are too messy to extract | After about 1.5 days, stop at Items plus numbered Notes, and fix the rest by hand |
| Colab offers no A100, or the compute units run low | Train on an L4 (about 8 hours a run, about 14 units). Buy a pay-as-you-go pack (100 units for $9.99) only if needed. The laptop steps cost nothing |
| Colab is unavailable altogether | Fall back on the school GPU (objectives plan, appendix) |
| The official code fails on Colab | Colab runs Linux, so it should run as the authors wrote it. Check `torch.cuda.is_available()` first, then read the error before changing code |
| Smart App Control blocks something on the laptop | Use python.org Python builds, never uv's. If a package is still blocked, run that step on Colab |
| The authors don't grant a licence | Keep their code in a separate private copy; publish only your own code and a list of changes |
| Training runs take longer than planned | Check the budget before the full run (at most about 8 hours). Questions are already capped per section. If it's over, cut in this order: a two-level ToC, epochs, then model size |
| vLLM won't install on Colab | Generate with `transformers` in 4-bit, batched; it's slower, so start question writing early |
| A strong untuned LLM given the ToC matches STAIR | Still a valid result. Make the case on local hosting, privacy, cost, speed and never naming a missing section |
| Colab ends a session mid-run (time limit, idle timeout, closed tab, sleeping laptop) | Keep the tab open and the laptop awake. Save checkpoints about every 30 minutes to Drive, and resume from the latest one instead of restarting |
| WRDS doesn't have a source the checklist needs | Use your own scores for that question, and say so |
| The fine-tuned model doesn't beat the simpler approaches | Still a valid result: report it honestly and explain why |
| Behind schedule | Cut the optional items (4.3, 2.3), then the whole-10-K method's synthetic sample, then objective 5's case studies and WRDS checks; never cut the core comparison or the write-up |

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
