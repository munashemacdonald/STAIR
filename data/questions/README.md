# Question files

## Templates for the hand-written test set

- `financebench_templates.jsonl`: the 25 generic analyst questions ("domain-relevant questions", dg01–dg25) from FinanceBench, rewritten as templates. Each line has:
  - `template`: the question with `{company}`, `{fy}` (the filing's fiscal year), `{fy_1}` and `{fy_2}` (one and two years earlier)
  - `financebench_original`: the question exactly as FinanceBench wrote it
  - `note`: where a template was adapted (two came from 10-Qs), or needs the 10-K cover page
- `checklist_templates.jsonl`: 8 more analyst questions (ck01–ck08). They were fixed on 2026-11-20, before any table of contents was parsed. They cover topics FinanceBench barely touches, such as supply chain, debt maturities, internal controls, cybersecurity and buybacks. Objective 5 reuses them as its checklist.

## The hand-written test set

`test_handwritten.jsonl` asks, of every company's FY2025 filing, held-out and test-only companies included (12 companies):
1. **7 FinanceBench templates:** dg09, dg11, dg14, dg19, dg20, dg22 and dg24.
   - Chosen on 2026-11-25, before any table of contents was parsed (`handwritten` in `config.yaml`).
   - dg09, dg11, dg20 and dg22 feed objective 5. dg14, dg19 and dg24 keep financial-statement questions in the set.
   - The other 18 stay in the file but aren't asked.
2. **all 8 checklist templates**
3. **36 company-specific questions:** 2 per seen company and 5 per unseen company (CRWV, FIG, ORCL, AMD), weighted toward the unseen companies on 2026-11-25.
   - Write all 36 before building the parser, from what an analyst would want to know (news, earnings calls). Each company's questions must be written before you see its table of contents, and building the parser shows you them.
   - Save them before you start labelling.
   - They're in `company_specific.jsonl`, drafted on 2026-11-26 before any filing was downloaded or parsed. They're merged into `test_handwritten.jsonl` when the set is built. A question the filing doesn't answer is kept, with the reason in `skipped`.

**No question names the company** (2026-11-25). Fill `{company}` in the templates with "the company", and write the company-specific questions the same way. The `company` field records which filing a question is about. Generated questions follow the same rule.

That gives up to 216 questions, about 200 after skips. It was cut from about 450 on 2026-11-25, because labelling 450 would take 25–40 hours. See step 1.9 of `docs/OBJECTIVES_PLAN.md`.

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

A method can pick a different section that also answers a question. To find those sections fairly, after every method has run, each method's top 3 sections for each test question are pooled and judged blind, without saying which method found them (step 3.7 of the objectives plan):
- **The judge** is the question-writing model (Qwen2.5-14B, temperature 0), for generated and hand-written questions alike. For hand-written questions it's given the labelling rule above.
- **You check a random 50** of its decisions on hand-written questions, blind, and report the agreement.

Sections judged correct at this stage go in **`pooled_answers_alone`**, kept separate from the labels made before any method ran. That way, results can be reported both with and without pooling.

## Line format

`test_handwritten.jsonl`, one question per line:

```
{"id": "NVDA_FY2025_dg14", "template": "dg14", "company": "NVDA", "fy": 2025,
 "group": "seen_company",
 "question": "Does the company have an improving operating margin profile as of FY2025? ...",
 "answers_alone": ["<section id>"], "needed_together": [], "pooled_answers_alone": [],
 "skipped": null}
```

- **Checklist questions** use their `ck` id as `template`.
- **Company-specific questions** use `"template": "custom"`, with ids like `NVDA_FY2025_cs01`.
- **`group`** is `seen_company` (the 8 training companies), `first_time` (CRWV, FIG) or `established` (ORCL, AMD).

**Generated questions** use the same fields. Since 2026-11-25 they are written per section and split by year:
- **`train`:** the 8 training companies' FY2024 filings, 3–6 questions per section, about 5,000 in all. Each has exactly one section in `answers_alone`, and no `group`.
- **`dev` and `test_synthetic`:** the same companies' FY2025 filings, 2 per section, with 20% of sections in dev and 80% in test.
- **`test_heldout`:** the 4 held-out and test-only companies' FY2025 filings, 2 per section.

In dev and test files, `answers_alone` holds the section the question was written from. Other correct sections are added later, in `pooled_answers_alone`.

## How these are scored

See step 3.1 of `docs/OBJECTIVES_PLAN.md`:
- **R@k:** a question counts as found if any correct section is in the top k. "Correct" means `answers_alone` plus `pooled_answers_alone`; scores are also reported without the pooled sections.
- **Needed-together questions** are left out of R@k and reported in one line: whether all the needed sections are in the top 3.
- **Headline:** the hand-written set is the headline result, split into two co-primary groups:
  - **temporal zero-shot:** the 8 seen companies, about 31 independent units
  - **cross-company zero-shot:** the 4 unseen companies, about 35 units

  Within each group, confidence intervals resample whole templates (15) and company-specific questions (each one its own unit), not single questions.
