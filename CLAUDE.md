# STAIR for 10-K Filings

A standalone project that applies STAIR ([Kumar et al., IBM, arXiv:2609.03874](https://arxiv.org/abs/2609.03874)) to SEC 10-K filings. STAIR is a fine-tuned LLM that retrieves by picking the right section from a document's table of contents. The project tests whether it beats standard RAG retrieval on 10-Ks, then shows how its cited answers can become signals for investment research. Timeline: Nov 2–30, 2026, at about 40 hours a week.

## Where things are

- `docs/PLAN.md`: the dated plan. It is the source of truth for scope decisions.
- `docs/OBJECTIVES_PLAN.md`: steps and expected results for each objective. `OBJECTIVES_PLAN.pdf` is generated from it.
- `docs/STAIR_Project_Overview.pdf`: one-page topic, goal and objectives.
- `config.yaml`: companies, held-out companies, fiscal years, models, training settings, seed.
- `README.md`: layout, setup and the status checklist.

## Decisions so far

- **Scope:** retrieval accuracy and building signals. No stock-price prediction or returns backtest; that is future work.
- **Only STAIR is trained (Nov 9).** The comparisons are untrained: BM25, embedding retrieval, an untuned LLM given the ToC, and an LLM given the whole 10-K. The paper's STAIR vs. DSI result is cited, not reproduced.
- **Companies (Nov 17):** Technology sector.
  - Training companies, in pairs: chips (NVDA, AVGO), internet (GOOGL, META), software (CRM, ADBE), hardware (AAPL, DELL).
  - Held out: CRWV and FIG. Both listed in 2025 and have only an FY2025 10-K, so the held-out test is about first-time filers. They were chosen by filing availability, before any results existed.
  - ORCL was dropped to make room for FIG.
- **Fiscal years:** each company's own FY2024 and FY2025, so "FY2025" ends anywhere from Jan to Dec 2025 depending on the company. That makes 18 filings in total.
- **Section targets** are numbered by position plus title, e.g. "7.3 Liquidity and Capital Resources", so every target is unique.
- **No official code:** the paper's code and data link has expired, so everything is rebuilt from the paper.

**Known gaps in the docs:** `OBJECTIVES_PLAN` still says 20 filings and recommends December fiscal years. `config.yaml` and the list above are correct.

## Environment

- Training runs on a separate GPU machine; its model, VRAM and OS aren't recorded yet. The user's laptop has no NVIDIA GPU, and cloud sessions have no GPU either.
- vLLM needs Linux or WSL.

## Rules

- Never commit anything under `data/raw/`, `data/parsed/`, `checkpoints/` or `papers/`; they are gitignored. Do commit `data/questions/`, especially the hand-written questions, which can't be regenerated.
- **SEC EDGAR:**
  - Every request needs a User-Agent containing a name and email, and must stay under 10 requests per second.
  - Read the identity from the `SEC_IDENTITY` environment variable. Don't commit a real email.
- When `docs/OBJECTIVES_PLAN.md` changes, regenerate its PDF or say that the PDF is out of date.
- The user is a financial engineering student. Explain technical terms the first time they come up, and ask before committing or pushing.
