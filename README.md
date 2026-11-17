# STAIR for 10-K Filings

Structure-aware retrieval for investment research: applying STAIR ([Kumar et al., IBM, arXiv:2609.03874](https://arxiv.org/abs/2609.03874)) to SEC 10-K filings.

STAIR is a fine-tuned LLM that retrieves information by picking the right section from a document's table of contents, instead of searching text chunks the way standard retrieval-augmented generation (RAG) does. This project tests whether STAIR finds the section of a 10-K that answers an analyst's question more accurately than standard RAG retrieval. It then shows how STAIR's cited answers can be turned into signals for investment research. It does not predict stock prices.

## Documents

- [Project overview](docs/STAIR_Project_Overview.pdf): topic, goal and objectives on one page
- [How to complete each objective](docs/OBJECTIVES_PLAN.md) ([PDF](docs/OBJECTIVES_PLAN.pdf))
- [Dated plan](docs/PLAN.md)
- [Architecture slides](docs/STAIR_Architecture.pptx)

## Layout

```
config.yaml        companies, fiscal years, models, training settings, seed
data/raw/          downloaded 10-K HTML (not committed; re-downloaded by src/)
data/parsed/       toc.json + content.jsonl per filing (not committed; rebuilt by the parser)
data/questions/    train / dev / test question files, including the hand-written ones
src/               parser, question generation, retrievers, training, scoring
checkpoints/       LoRA adapters (not committed)
results/           result tables (CSV) and figures
docs/              plans, overview and slides
papers/            reading-list PDFs (not committed)
```

## Setup

On the GPU machine:

1. Install PyTorch for the machine's CUDA version, following the instructions on [pytorch.org](https://pytorch.org).
2. Run `pip install -r requirements.txt`. Everything runs natively on Windows; vLLM is not used because it needs Linux.
3. Fill in `config.yaml`: the sector, the companies, and your name and email for SEC EDGAR. The SEC requires these in every request.

## Status

- [x] Plan, objectives and slides (`docs/`)
- [ ] Objective 1: build the 10-K question dataset
- [ ] Objective 2: train STAIR
- [ ] Objective 3: compare STAIR with the alternatives
- [ ] Objective 4: test on new filings
- [ ] Objective 5: show the investment use
- [ ] Objective 6 (stretch): scale to 25–30 companies
