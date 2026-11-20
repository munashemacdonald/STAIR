# Question files

## Templates for the hand-written test set

- `financebench_templates.jsonl`: the 25 generic analyst questions ("domain-relevant questions", dg01–dg25) from FinanceBench, rewritten as templates. Each line has:
  - `template`: the question with `{company}`, `{fy}` (the filing's fiscal year), `{fy_1}` and `{fy_2}` (one and two years earlier)
  - `financebench_original`: the question exactly as FinanceBench wrote it
  - `note`: where a template was adapted (two came from 10-Qs), or needs the 10-K cover page
- `checklist_templates.jsonl`: 8 more analyst questions (ck01–ck08). They were fixed on 2026-11-20, before any table of contents was parsed. They cover topics FinanceBench barely touches, such as supply chain, debt maturities, internal controls, cybersecurity and buybacks. Objective 5 reuses them as its checklist.

## The hand-written test set

`test_handwritten.jsonl` asks, of every company's FY2025 filing, held-out companies included:
1. **all 25 FinanceBench templates**
2. **all 8 checklist templates**
3. **5 company-specific questions per company:**
   - Write these before opening that company's table of contents, from what an analyst would want to know (news, earnings calls).
   - Save them before you start labelling.

That gives up to about 380 questions. See step 1.9 of `docs/OBJECTIVES_PLAN.md`.

**Source and licence:** Islam et al., "FinanceBench: A New Benchmark for Financial Question Answering" (2023), dataset `PatronusAI/financebench`, licensed CC BY-NC 4.0. Credit FinanceBench, and keep any question set built from its templates non-commercial.

**Keep these questions away from training.**
- Never put them, or the templates, in the question generator's prompt.
- Generated questions that closely match any of them are dropped (step 1.7).

## Labelling rule

A 10-K repeats facts across Items, so a question can have more than one correct section. Label before looking at any method's output, and record two lists of section ids (the `id` values in the filing's `toc.json`):

- **`answers_alone`:** each of these sections answers the question by itself. Any one of them is enough.
- **`needed_together`:** these sections answer the question only together, and none of them does alone. For example, a ratio whose inputs sit in the balance sheet and a note.

A section **doesn't count** if it:
- only points elsewhere ("see Note 12")
- mentions the topic in general terms, like a boilerplate risk factor about litigation
- only supports an answer another section gives. For "What drove operating margin change?", MD&A explains the drivers, so the income statement doesn't count.

If a question doesn't apply to a company, keep the line, leave both lists empty, and put the reason in `skipped`.

**Second labeller.** Before labelling everything, a classmate labels about 30 questions with this rule, without seeing your labels:
- Pick the 30 at random, spread across templates and companies.
- Compare your labels with theirs, settle the disagreements, and tighten this rule where it was unclear. Then label the rest.
- Report the agreement in the write-up.

If nobody is available, re-label the same 30 yourself at least three days later, without looking at your first labels. That's weaker, but better than nothing.

## Pooling: sections added after the methods run

Step 1.8 checks only the sections that BM25 and embeddings suggest. On its own, that would let those two methods get credit for an alternative correct section while other methods' alternatives were never checked.

So after every method has run, each method's top 3 sections for each test question are pooled and judged blind, without saying which method found them (step 3.6 of the objectives plan):
- **Generated test questions:** the LLM judge from step 1.8 checks them.
- **Hand-written questions:** you check them.

Sections judged correct at this stage go in **`pooled_answers_alone`**, kept separate from the labels made before any method ran. That way, results can be reported both with and without pooling.

## Line format

`test_handwritten.jsonl`, one question per line:

```
{"id": "NVDA_FY2025_dg17", "template": "dg17", "company": "NVDA", "fy": 2025,
 "question": "What drove operating margin change as of FY2025 for NVIDIA?",
 "answers_alone": ["<section id>"], "needed_together": [], "pooled_answers_alone": [],
 "skipped": null}
```

- **Checklist questions** use their `ck` id as `template`.
- **Company-specific questions** use `"template": "custom"`, with ids like `NVDA_FY2025_cs01`.

**Generated questions** (`train`, `dev`, `test_synthetic`, `test_heldout`) use the same fields. `answers_alone` starts with the section the question was written from, plus any other section the judge in step 1.8 says also answers it.

## How these are scored

See step 3.1 of `docs/OBJECTIVES_PLAN.md`:
- **R@k:** a question counts as found if any correct section is in the top k. "Correct" means `answers_alone` plus `pooled_answers_alone`; scores are also reported without the pooled sections.
- **Needed-together questions** are scored separately: whether all the needed sections are in the top 3.
- **Headline:** the hand-written set is the headline result. Its confidence intervals resample whole templates (33 of them) and company-specific questions (each one its own unit), not single questions.
