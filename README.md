# Deal With It! Diagnosing Post-Training in Bargaining, from Fluency to Surplus Extraction

Code, archived bargaining episodes, reference calculations, and companion figures for Mohid Butt's paper, accepted at **SLM-Agents (SLMs for Agentic Systems) and FAST, NeurIPS 2026 workshops**.

[Paper and reviews on OpenReview](https://openreview.net/forum?id=m1gDHXY8ST) · [Offline companion figures](figures/index.html) · [Reproduction checks](scripts/reproduce.py)

Two players alternate integer offers, accept a standing offer, or walk away. The environment computes a zero-sum reward from the agreed price and both players' reservation values. It supports scripted opponents, an exact best response with privileged opponent information, and a searched history-blind offer sequence with standing-offer-reactive acceptance.

## Quick start

Requires Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and Node.js 20 or newer for SVG export. The commands below run locally without API keys or GPUs.

CI uses Python 3.12, uv 0.11.3, and Node.js 22.

```sh
git clone https://github.com/mohidbt/bargaining-post-training.git
cd bargaining-post-training
uv sync --locked
uv run pytest
uv run python scripts/reproduce.py
```

Open [figures/index.html](figures/index.html) in a browser to explore the figures. It is a self-contained page that works offline.

Installation downloads dependencies. After installation, the tests and reproduction checks use the included data without contacting model providers. Node.js is needed to rebuild and check the SVG exports; viewing the existing figures only requires a browser.

Successful reproduction ends with:

```text
Source hashes, all figures, step-50 replay, and both frontier comparisons verified.
```

The reproduction script regenerates files in `figures/`. On an unchanged checkout, `git diff --exit-code` should report no changes afterward. [GitHub Actions](.github/workflows/checks.yml) runs the locked install, tests, reproduction, and this check.

## Find and reproduce a result

Run commands from the repository root. These calculations use a CPU and require no API keys or GPUs.

Create the directory for optional comparison outputs first:

```sh
mkdir -p results
```

| Result or check | Command | Included evidence |
| --- | --- | --- |
| Training curves, opening offers, archived step-50 replay, and companion figures | `uv run python scripts/reproduce.py` | [Training archive](data/training/), [figure data](figures/data.json), [source hashes](figures/sources.json) |
| Frontier comparison with the searched comparator, including confidence intervals and surplus share | `uv run python scripts/paired_vs_blind.py --out results/vs-blind.json` | [Frontier episodes](data/frontier/), [stored comparison](data/references/vs-blind.json) |
| Frontier comparison with the privileged upper bound | `uv run python scripts/paired_vs_upper.py --out results/vs-upper.json` | [Frontier episodes](data/frontier/), [stored comparison](data/references/vs-upper.json) |
| Larger comparator search and upper bound on identical refinement episodes | `uv run python scripts/bounds_on_refine.py --out results/bounds` | [Stored bounds](data/references/bounds.json) |

The quick-start reproduction command also verifies both frontier comparisons against their stored results. The larger search is separate and takes longer; it writes `bounds.json` and `bounds.out` under `results/bounds/`.

## What is included

| Directory | Contents |
| --- | --- |
| `market/` | Bargaining mechanics, scripted opponents, seeded datasets, prompts, exact best response, comparator search, and Verifiers adapter |
| `scripts/` | Offline reference comparisons and reproduction checks |
| `data/training/` | Training run two's plotted metrics and archived rollout pages at steps 1, 5, 25, and 50 |
| `data/frontier/` | 64 archived exploratory frontier episodes for each role and sampling provenance |
| `data/references/` | Computed reference values and comparisons on identical frontier episodes |
| `figures/` | Interactive figures, their source data, and eight SVG exports |
| `configs/training.toml` | Recorded hosted training settings, with local environment identifiers |
| `tests/` | Environment, opponent, parser, replay, dataset, and evaluation checks |

The archived training pages contain 64 of the 512 rollouts at each selected step. Repeated setups are retained. Zero-advantage percentages are sample shares from the full logged batch, rather than estimates from those archived pages. At step 50 the logged shares are 64.3% for the seller and 60.8% for the buyer.

Training data is exported with only fields needed for the figures and replay. Service run identifiers, timestamps, timing telemetry, and unrelated metrics have been removed. Prompts, model completions, game parameters, seeds, rewards, and plotted measurements are preserved. File hashes in [figures/sources.json](figures/sources.json) refer to these public exports.

## Reproduce the references

The quick-start reproduction checks rebuild every figure, reconstruct all 64 archived step-50 rollouts, and recompute both frontier comparisons with the archived bootstrap settings. It verifies the resulting numbers against the stored results.

To repeat the larger comparator search and the exact best-response calculation on 8,000 episodes for each role:

```sh
uv run python scripts/bounds_on_refine.py --out results/bounds
```

This runs on a CPU and takes longer than the quick-start checks. The pooled midpoint rewards are approximately 0.3502 for the searched comparator and 0.5433 for the full information upper bound, on identical refinement episodes. The search considers 2,616 candidates in its screening stage and refines the leading 20 on disjoint seeds.

The upper bound knows the opponent's hidden class and parameters. The learner does not. The searched comparator is a lower bound on the best attainable performance among history-blind policies. Exceeding it would not by itself establish inference from opponent behavior.

The frontier comparison is exploratory: one preview model accessed 2026-08-05, 64 episodes for each role, with provider default reasoning effort. The recorded `gpt-5.6-terra` identifier is not pinned to a dated snapshot. The trained 4B model has no frozen evaluation suitable for this reference comparison.

## Training integration

`market.vf_env.load_environment` and `rlvr_bargain.load_environment` expose the environment to Verifiers. The recorded settings use Qwen3.5-4B, 16 rollouts for each example, a batch size of 512, and up to 768 generated tokens. They document the hosted experiment; a training backend and model weights are required to train again. The offline reproduction commands do not start training or contact model providers.

This release reproduces calculations from archived outputs. It does not include a trained checkpoint, a complete training backend, or a script to collect new frontier responses. The companion figures visualize the released evidence; they are not a rebuild of the paper's typeset PDF. The lockfile fixes the released analysis environment, rather than reconstructing every dependency of the original hosted training service.

## Code navigation

- [Environment and reward](market/env.py): actions, turn order, termination, and midpoint reward.
- [Scripted opponents](market/bots.py) and [datasets and prompts](market/vf_env.py): opponent behavior and seeded game generation.
- [Exact best response](market/best_response.py): the reference with privileged opponent information.
- [Comparator search](market/blind_search.py): candidate policies and the search protocol.
- [Recorded training configuration](configs/training.toml): settings for the archived experiment.

## Citation

```bibtex
@misc{butt2026dealwithit,
  author = {Mohid Butt},
  title = {Deal With It! Diagnosing Post-Training in Bargaining, from Fluency to Surplus Extraction},
  year = {2026},
  note = {Accepted at SLM-Agents (SLMs for Agentic Systems) and FAST, NeurIPS 2026 Workshops},
  url = {https://openreview.net/forum?id=m1gDHXY8ST}
}
```

## License

The code and accompanying data are available under the [MIT License](LICENSE). External packages and model weights retain their own licenses.
