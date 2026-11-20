# Question files

- `financebench_templates.jsonl`: the 25 generic analyst questions ("domain-relevant questions", dg01–dg25) from FinanceBench, rewritten as templates. Each line has:
  - `template`: the question with `{company}`, `{fy}` (the filing's fiscal year), `{fy_1}` and `{fy_2}` (one and two years earlier)
  - `financebench_original`: the question exactly as FinanceBench wrote it
  - `note`: where a template was adapted (two came from 10-Qs), or needs the 10-K cover page
- The hand-written test set (`test_handwritten.jsonl`) asks these of every company's FY2025 filing and labels every section that answers each one. See step 1.9 of `docs/OBJECTIVES_PLAN.md`.

**Source and licence:** Islam et al., "FinanceBench: A New Benchmark for Financial Question Answering" (2023), dataset `PatronusAI/financebench`, licensed CC BY-NC 4.0. Credit FinanceBench, and keep any question set built from these templates non-commercial.

## Labelling rule

A 10-K repeats facts across Items, so a question can have more than one correct section. Label before looking at any method's output, and record two lists of section ids (the `id` values in the filing's `toc.json`):

- **`answers_alone`:** each of these sections answers the question by itself. Any one of them is enough.
- **`needed_together`:** these sections answer the question only together, and none of them does alone. For example, a ratio whose inputs sit in the balance sheet and a note.

A section **doesn't count** if it:
- only points elsewhere ("see Note 12")
- mentions the topic in general terms, like a boilerplate risk factor about litigation
- only supports an answer another section gives. For "What drove operating margin change?", MD&A explains the drivers, so the income statement doesn't count.

If a template doesn't apply to a company, keep the line, leave both lists empty, and put the reason in `skipped`.

## Line format

`test_handwritten.jsonl`, one question per line:

```
{"id": "NVDA_FY2025_dg17", "template": "dg17", "company": "NVDA", "fy": 2025,
 "question": "What drove operating margin change as of FY2025 for NVIDIA?",
 "answers_alone": ["<section id>"], "needed_together": [], "skipped": null}
```

Generated questions (`train`, `dev`, `test_synthetic`, `test_heldout`) use the same two fields. `answers_alone` starts with the section the question was written from, plus any other section the judge in step 1.8 says also answers it.

## How these are scored

See step 3.1 of `docs/OBJECTIVES_PLAN.md`:
- **R@k:** a question counts as found if any `answers_alone` section is in the top k.
- **Needed-together questions** are scored separately: whether all the needed sections are in the top 3.
