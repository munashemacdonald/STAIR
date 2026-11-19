# STAIR for 10-K Filings

Structure-aware retrieval for investment research: applying STAIR ([Kumar et al., IBM, arXiv:2609.03874](https://arxiv.org/abs/2609.03874)) to SEC 10-K filings.

STAIR is a fine-tuned LLM that retrieves information by picking the right section from a document's table of contents, instead of searching text chunks the way standard retrieval-augmented generation (RAG) does. This project tests whether STAIR finds the section of a 10-K that answers an analyst's question more accurately than standard RAG retrieval. It then shows how STAIR's cited answers can be turned into signals for investment research. It does not predict stock prices.

Training builds on the authors' official implementation, [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean) (commit `3729c54`), adapted for 10-K filings. Every change to it is listed in the write-up.

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

On the GPU machine (Windows, Python 3.15.2, no admin rights needed). The full steps are in [docs/OBJECTIVES_PLAN.md](docs/OBJECTIVES_PLAN.md), under "Before you start".

1. Install `uv` for your user: `py -3.15 -m pip install --user uv`.
2. Create this repo's environment and install PyTorch's **CUDA** build. A plain install on Windows gets the CPU-only build.
   ```
   uv venv --python 3.15
   uv pip install torch --index-url https://download.pytorch.org/whl/cu130
   uv pip install -r requirements.txt
   ```
3. Check the GPU: `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` should print `True NVIDIA RTX 4000 SFF Ada Generation`.
4. Set these user environment variables. Never write their values into a file in this repo.
   - `SEC_IDENTITY`: your name and email. The SEC requires these in every request.
   - `HF_TOKEN`: your Hugging Face token.
   - `HF_HOME`: a folder with about 60 GB free, for downloaded models.
   - `PYTHONUTF8=1`
5. Set up your fork of the official STAIR code next to this repo, as described in the objectives plan: CUDA index, `bitsandbytes`, no `torchrun`. Then run its bundled example.

Everything runs natively on Windows. vLLM is not used because it needs Linux.

## Status

- [x] Plan, objectives and slides (`docs/`)
- [ ] Objective 1: build the 10-K question dataset
- [ ] Objective 2: train STAIR
- [ ] Objective 3: compare STAIR with the alternatives
- [ ] Objective 4: test on new filings
- [ ] Objective 5: show the investment use
- [ ] Objective 6 (stretch): scale to 25–30 companies
