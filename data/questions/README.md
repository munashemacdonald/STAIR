# Question files

- `financebench_templates.jsonl`: the 25 generic analyst questions ("domain-relevant questions", dg01–dg25) from FinanceBench, rewritten as templates. Each line has:
  - `template`: the question with `{company}`, `{fy}` (the filing's fiscal year), `{fy_1}` and `{fy_2}` (one and two years earlier)
  - `financebench_original`: the question exactly as FinanceBench wrote it
  - `note`: where a template was adapted (two came from 10-Qs), or needs the 10-K cover page
- The hand-written test set (`test_handwritten.jsonl`) asks these of every company's FY2025 filing and labels every section that answers each one. See step 1.9 of `docs/OBJECTIVES_PLAN.md`.

**Source and licence:** Islam et al., "FinanceBench: A New Benchmark for Financial Question Answering" (2023), dataset `PatronusAI/financebench`, licensed CC BY-NC 4.0. Credit FinanceBench, and keep any question set built from these templates non-commercial.
