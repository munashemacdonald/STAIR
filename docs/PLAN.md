# STAIR for 10-K filings: plan

At about 40 hours a week.
Based on: STAIR, arXiv 2609.03874 (`papers/Alternate STAR.pdf`, kept out of git).
- The authors' code is public as of the paper's v2: [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean).
- The project builds on that code and adapts it for 10-Ks. CLAUDE.md lists the changes it needs.

This checklist follows the same objectives as `OBJECTIVES_PLAN.md` and the project overview. The numbers in brackets, like (1.8), are the steps in the objectives plan, where each item is explained.

**The core test:** does STAIR, trained on FY2024 tables of contents, find the right section of an FY2025 10-K more often than the best standard RAG pipeline (hybrid + reranker)? It's measured for:
- **companies seen in training:** temporal zero-shot
- **companies never seen:** cross-company zero-shot

This is the question the paper leaves open in its Section 7. Objective 5 then demonstrates the investment use.

## Order of work

Re-planned on Nov 25 with about 15 days left, and cut to the core test. "What's in, and what's left out" in the objectives plan lists everything that was cut.

1. Setup, then objectives 1, 2, 3, 4 and 5, then the write-up. About 13.25 days, leaving about 1.75 days of slack (four extra test companies were added on Nov 26).
2. Nothing is optional any more: anything outside the core test is future work.

**Out of scope:** training any baseline, the analyst agent itself, predicting stock prices and testing trading returns.

---

## Setup
- [ ] Read the 7 core papers (see the reading list at the bottom)
- [x] Choose a sector and companies: Technology; 8 training companies, 2 held-out first-time filers (CRWV, FIG) and 6 established test-only companies (ORCL, AMD; FFIV, AKAM, SWKS and JKHY added Nov 26, chosen by the author), listed in `config.yaml`
- [x] Choose where the GPU work runs (Nov 24): Google Colab, paid through the Google AI Pro plan (about 200 compute units a month). The laptop runs every step that needs no GPU. The school GPU (RTX 4000 SFF Ada, 20 GB, Windows) is the fallback.
- [x] Laptop environment (Nov 24): `.venv` on the python.org Python 3.14.0, with CPU-only torch, `requirements.txt` and `ipykernel`. Smart App Control blocks uv's own Python builds, and `ranx` has no Windows builds for 3.15.
- [x] Give this project its own git repo: [github.com/munashemacdonald/STAIR](https://github.com/munashemacdonald/STAIR)
- [x] Set `SEC_IDENTITY`, `HF_TOKEN` and `PYTHONUTF8=1` on the laptop (Nov 26; the token works)
- [ ] Install Google Drive for desktop, and make `My Drive/stair/` with `parsed/`, `questions/`, `checkpoints/`, `runs/` and `logs/`
- [ ] Check Colab: the compute-unit balance shows about 200, and L4 and A100 are offered under Runtime → Change runtime type
- [ ] Add `HF_TOKEN` and a read-only `GITHUB_TOKEN` to Colab Secrets
- [ ] Make the Colab notebook, `notebooks/colab_gpu.ipynb`: mount Drive, read the tokens, clone the repos, install, and check the GPU. Commit it with outputs cleared.
- [ ] Make a private copy of the official STAIR code. A GitHub fork would be public, and the code has no licence. Add `bitsandbytes`, then run its bundled example end to end on a Colab GPU.

**Done when:** the official code's bundled example trains and evaluates on a Colab GPU.

---

## Objective 1: Build a 10-K question dataset

**First, before building the parser**
- [x] Write the 56 company-specific questions from news and earnings calls: 2 per seen company and 5 per unseen company. They say "the company", never its name. Done Nov 26 in `data/questions/company_specific.jsonl`, each committed before that company's filing was downloaded or parsed. (1.9)

**Filings → tables of contents and section text**
- [x] Download 24 10-Ks from EDGAR with `edgartools`, sending a User-Agent with your contact (1.1), using step 1.1 in `notebooks/objective1_dataset.ipynb`; the filings are listed in `data/filings.csv`. All 24 downloaded on Nov 26.
  - FY2024 and FY2025 for the 8 training companies
  - FY2025 only for the 2 held-out and 6 test-only companies
- [x] Split each filing into Items (1.2). Done Nov 26: 309 Items across the 24 filings, all passing the safety check. Two fixes: when Item 8 only points to Item 15 (NVDA, ORCL), Item 15 is kept because it holds the financial statements; and text at 16pt or more counts as a heading, because FIG sets headings by size rather than bold.
- [x] Pull out the headings inside the Items, including the notes to the financial statements (1.3). Done Nov 26: 2,935 sections across the 24 filings, in `data/parsed/<filing>/outline.jsonl`. A heading is a whole bold line of at most 12 words; Items 8 and 15 have three levels (statements and Notes, then the headings inside each Note), other Items two. Page headers, exhibit lists and the risk factor summary's repeated headings are removed, and sections under 30 words are joined with the next. FFIV's individual risks (bold italics, lost in the Markdown) are left for the hand check (1.6).
- [ ] Save per filing: `toc.json` in the official code's format, with the title "SEC Form 10-K filing", and `content.jsonl` (section → text) (1.4–1.5)
- [ ] Check and fix each table of contents by hand, and record how many fixes each needed (1.6)

**Checkpoint:** heading extraction is time-boxed to about 1.5 days. If headings inside Items are still a mess by then, stop at Items plus numbered Notes and move on. (1.3)

**Generated questions**
- [ ] Write questions per section with Qwen2.5-14B-Instruct-AWQ, served with vLLM on a Colab L4. Each question is labelled with the one section it came from and says "the company", never the name. (1.7)
  - FY2024, 8 training companies: 3–6 per section, about 5,000 for training
  - FY2025, all 16 companies: 2 per section, for dev and test
- [ ] Keep the hand-written questions out of generation: never use them or their templates as prompt examples, and drop generated questions that closely match them (cosine above about 0.85) (1.7)
- [ ] Remove duplicates, drop questions their passage doesn't answer, and read 100 by hand (1.8)
- [ ] Copy the questions from Drive into `data/questions/` and commit them

**Hand-written test set: the headline test** (1.9)
- [ ] Ask these of every company's FY2025 filing, with `{company}` filled as "the company". That's 16 companies and about 280 questions:
  - 7 FinanceBench templates: dg09, dg11, dg14, dg19, dg20, dg22, dg24
  - the 8 checklist templates
  - the 56 company-specific questions
- [ ] Second labeller: a classmate labels about 30 random questions with the rule in `data/questions/README.md`, without seeing your labels. Compare, settle disagreements, tighten the rule, then label the rest.

**Splits, by year** (1.10)
- [ ] `train.jsonl`: every FY2024 question of the 8 training companies
- [ ] The 8 training companies' FY2025 questions, split by section: 20% to `dev.jsonl`, 80% to `test_synthetic.jsonl`
- [ ] The 8 unseen companies' questions in `test_heldout.jsonl`
- [ ] Every FY2025 question has a `group` field: `seen_company`, `first_time` or `established`

**Done when:** 24 clean tables of contents; train, dev and test files; the hand-written set labelled, with the second-labeller agreement.

---

## Objective 2: Train STAIR on the filings
- [ ] Add 4-bit (QLoRA) loading to the official code, so Mistral-7B fits on an L4 (24 GB). Use it on every GPU. (2.2)
- [ ] Adapt the official code for 10-Ks: each question paired with its own filing's ToC, the "SEC Form 10-K filing" header with no company name or year, and a longer maximum length (2.3)
- [ ] Add a `--resume` option to your copy's `train` command, and test it once (2.4)
- [ ] Estimate the training time from a 200-step timing: at most about 8 hours and 5 epochs. If it's over, cut in this order: a two-level ToC, fewer epochs, a smaller model. (2.4)
- [ ] Train on the FY2024 questions on Colab, with the tab open and the laptop awake. Save a checkpoint about every 30 minutes to Google Drive and keep every one. Record the GPU type and units used. (2.4)
- [ ] Choose the checkpoint by dev R@1, scoring every checkpoint on the 300-question dev sample (2.5)
- [ ] Decode with 5-beam search, no constraint: R@1 from the top beam, the top 3 from the beams, invalid titles counted (2.6)

**Done when:** the checkpoint is chosen by dev R@1, and STAIR returns a top 3 for any question.

---

## Objective 3: Compare STAIR with standard RAG
None of these methods is trained. Every method is tested on the FY2025 filings.

- [ ] Scoring script: R@1 and R@3 (any correct section is in the top k), plus All@3 for needed-together questions; tested on a made-up run (3.1)
- [ ] Chunks for every RAG method: about 300 words, overlapping by 50, each starting with its section number and title. A section scores as its best chunk (MaxP). (3.2)
- [ ] BM25 (3.2), embeddings with NV-Embed-v2 (3.3), and the hybrid of the two with reciprocal rank fusion (3.4)
- [ ] Hybrid + reranker, `BAAI/bge-reranker-v2-m3` over the hybrid's top 20 sections: the row to beat (3.5)
- [ ] The untuned Mistral given the table of contents, with STAIR's prompt and decoding (3.6)
- [ ] Pooling: the LLM judge checks each method's top 3 sections that aren't labelled yet, without knowing which method found them. Check 50 of its hand-written decisions yourself, blind. Rescore with and without the pooled sections. (3.7)
- [ ] Statistics (3.8):
  - **two co-primary comparisons,** fixed in `config.yaml` before any test: STAIR against hybrid + reranker, R@1 on the hand-written set, for seen companies and for unseen companies. Report the confidence interval of the difference, resampling templates and company-specific questions as units, with Holm across the two.
  - **secondary:** every other comparison, including the synthetic held-out set, Holm-adjusted
  - **freeze the test sets:** make every choice on dev
- [ ] Cost and speed for each method, in seconds per question and Colab units (3.9)
- [ ] Results by group: seen companies, first-time filers, established companies (3.10)

**Done when:** the main results table exists, after pooling, with both co-primary confidence intervals.

---

## Objective 4: Test whether it holds up on new filings
- [ ] Label each seen company's FY2025 section as unchanged, renamed or new (4.1)
- [ ] Break down STAIR's and hybrid + reranker's accuracy by that label, showing the counts, and count FY2024 titles among STAIR's invalid outputs (4.2)
- [ ] Add the unseen companies, per company (4.3)

**Done when:** you can say where STAIR goes wrong on next year's filing.

---

## Objective 5: Show the investment use
- [ ] Write a 1–5 scale in plain words for each of the 12 checklist questions (5.1)
- [ ] Run the checklist for every company and year with STAIR's top 3 sections, each capped at about 3,000 tokens. The answer comes with a score, a quote and the cited section. (5.2)
- [ ] Add the year-over-year score change (5.3)
- [ ] Check 20 answers by hand: score, section, quote and citation (5.4)

**Done when:** the signal table, the heat map and the hand-check numbers exist.

---

## Write-up
- [ ] Results: the main table, the two co-primary confidence intervals, and the new-filings breakdown
- [ ] Mistakes: a quick look at a sample of STAIR's misses, separating a wrong section from a section that doesn't exist
- [ ] Frame the results against the paper. It trains and tests STAIR in-corpus, so its 82.6% isn't comparable; this project tests the zero-shot question its Section 7 leaves open, and either outcome is a finding. Note that this project follows the official code's list output, not the paper's Table 2.
- [ ] Limitations:
  - STAIR is the only method trained on these companies' filings (FY2024); the RAG methods index FY2025 directly
  - dev holds seen companies only, so the checkpoint is tuned for temporal zero-shot
  - company names are removed, but a table of contents can still reveal the company
  - only 2 first-time filers; the seen group has only about 31 independent units and the unseen group about 55
  - the four test companies added on Nov 26 were chosen by the author, not drawn at random
  - the project is non-commercial (FinanceBench and NV-Embed-v2 licences)
- [ ] In the methods section, describe the hand-written test set: 7 of FinanceBench's 25 domain-relevant questions (Islam et al., 2023), the 8 checklist questions and 56 company-specific ones, adapted to each company's FY2025 10-K with names replaced by "the company". Say how many were skipped and why, and report the second-labeller agreement and how many sections pooling added for each method.
- [ ] List what each part is licensed under (objectives plan, "Licences"), and record where each step ran (GPU type, units, Python versions)
- [ ] Future work: everything in "What's in, and what's left out" in the objectives plan

---

## Risks and what to do about each

| Risk | What to do |
|---|---|
| Headings inside the Items are too messy to extract | After about 1.5 days, stop at Items plus numbered Notes, and fix the rest by hand |
| Colab offers no A100, or the units run low | Train on an L4 (about 8 hours a run, about 14 units). Buy a pay-as-you-go pack (100 units for $9.99) only if needed |
| Colab is unavailable altogether | Fall back on the school GPU (objectives plan, appendix) |
| Colab ends a session mid-run | Keep the tab open and the laptop awake. Save checkpoints about every 30 minutes to Drive, and resume from the latest one |
| vLLM won't install on Colab | Generate with `transformers` in 4-bit, batched; it's slower, so start question writing early |
| Training takes longer than planned | Check the budget before the full run. If it's over: a two-level ToC, fewer epochs, then a smaller model |
| Smart App Control blocks something on the laptop | Use python.org Python builds, never uv's. If a package is still blocked, run that step on Colab |
| The authors don't grant a licence | Keep their code in a separate private copy; publish only your own code and a list of changes |
| STAIR doesn't beat hybrid + reranker | Still a valid result: report it honestly and explain why |

## Reading

**The 7 core papers:**
1. DSI: "Transformer Memory as a Differentiable Search Index" (Tay et al., 2022)
2. DSI++: "Updating Transformer Memory with New Documents" (Mehta et al., 2022)
3. LoRA (Hu et al., 2021)
4. QLoRA (Dettmers et al., 2023)
5. DPR: "Dense Passage Retrieval" (Karpukhin et al., 2020)
6. InPars (Bonifacio et al., 2022), for generating questions
7. FinanceBench (Islam et al., 2023)

**When you reach that point:**

| When | Read |
|---|---|
| Objective 1 | SEC "How to Read a 10-K" (Item structure) |
| Objective 3 | Reciprocal rank fusion (Cormack et al., 2009); randomization tests (Smucker et al., 2007) |
| Write-up | RAG (Lewis et al., 2020); "Lost in the Middle" (Liu et al., 2024) |
