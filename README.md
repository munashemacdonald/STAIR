# STAIR for 10-K Filings

Structure-aware retrieval for investment research: applying STAIR ([Kumar et al., IBM, arXiv:2609.03874](https://arxiv.org/abs/2609.03874)) to SEC 10-K filings.

STAIR is a fine-tuned LLM that retrieves information by picking the right section from a document's table of contents, instead of searching text chunks the way standard retrieval-augmented generation (RAG) does. This project tests whether STAIR finds the section of a 10-K that answers an analyst's question more accurately than standard RAG retrieval. It then shows how STAIR's cited answers can be turned into signals for investment research. It does not predict stock prices.

Training builds on the authors' official implementation, [stair-clean](https://github.com/meghanadhpulivarthi/stair-clean) (commit `3729c54`), adapted for 10-K filings. Every change to it is listed in the write-up.

## Documents

- [Project overview](docs/STAIR_Project_Overview.pdf): topic, goal and objectives
- [How to complete each objective](docs/OBJECTIVES_PLAN.md) ([PDF](docs/OBJECTIVES_PLAN.pdf))
- [Plan checklist](docs/PLAN.md)
- [Architecture slides](docs/STAIR_Architecture.pptx)

## Layout

```
config.yaml        companies, fiscal years, models, training settings, seed
data/raw/          downloaded 10-K HTML (not committed; re-downloaded by src/)
data/parsed/       toc.json + content.jsonl per filing (not committed; rebuilt by the parser)
data/questions/    train / dev / test question files, including the hand-written ones
src/               parser, question generation, retrievers, training, scoring
notebooks/         the Colab notebook that starts each GPU step (committed without outputs)
checkpoints/       LoRA adapters (not committed; kept on Google Drive)
results/           result tables (CSV) and figures
docs/              plans, overview and slides
papers/            reading-list PDFs (not committed)
```

## Setup

The work runs in two places:
- **The laptop** (Windows, no GPU) runs every step that needs no GPU.
- **Google Colab** runs every step that does, paid through a Google AI Pro plan.
- **Files move between them** through GitHub (code) and Google Drive (data, checkpoints, run outputs).

The full steps are in [docs/OBJECTIVES_PLAN.md](docs/OBJECTIVES_PLAN.md), under "Setting up the laptop and Colab".

**Laptop**
1. Create this repo's environment with the python.org Python 3.14. Smart App Control blocks uv's own Python builds, and `ranx` has no Windows builds for 3.15. In PowerShell:
   ```
   uv venv --python (py -3.14 -c "import sys; print(sys.executable)")
   uv pip install -r requirements.txt ipykernel
   ```
   A plain install gets CPU-only torch, which is what the laptop needs.
2. Set these user environment variables. Never write their values into a file in this repo.
   - `SEC_IDENTITY`: your name and email. The SEC requires these in every request.
   - `HF_TOKEN`: your Hugging Face token.
   - `PYTHONUTF8=1`
3. Install Google Drive for desktop, and make `My Drive/stair/`.

**Colab**
1. Check the compute-unit balance (Runtime → View resources), and choose an L4 or A100 (Runtime → Change runtime type).
2. Add `HF_TOKEN` and a read-only `GITHUB_TOKEN` to Colab Secrets.
3. Run `notebooks/colab_gpu.ipynb`. It mounts Drive, clones this repo and your private copy of the official STAIR code, installs `requirements.txt`, and checks the GPU. Colab already has CUDA torch; don't reinstall it.
4. When the session's work is done, use Runtime → Disconnect and delete runtime, so it stops using units.

The school GPU (Windows, Python 3.15.2) is the fallback; its setup is in the objectives plan's appendix.

## Status

- [x] Plan, objectives and slides (`docs/`)
- [ ] Objective 1: build the 10-K question dataset
- [ ] Objective 2: train STAIR
- [ ] Objective 3: compare STAIR with the alternatives
- [ ] Objective 4: test on new filings
- [ ] Objective 5: show the investment use
- [ ] Objective 6 (stretch): scale to 25–30 companies
