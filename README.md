# First Experiments in Bargaining Post-Training

Code, archived bargaining episodes, and reproducible figures for [the blog post](https://mohid.io/blog/bargaining-post-training) by Mohid Butt.

Two players alternate integer offers, accept a standing offer, or walk away. The environment computes a zero-sum reward from the agreed price and both players' reservation values. It supports scripted opponents, an exact best response with privileged opponent information, and a searched history-blind offer sequence with standing-offer-reactive acceptance.

## Quick start

Requires Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and Node.js 20 or newer for SVG export. The commands below run locally without API keys or GPUs.

```sh
uv sync --locked
uv run pytest
uv run python scripts/reproduce.py
```

Open [figures/index.html](figures/index.html) in a browser to explore the figures. It is a self-contained page that works offline.

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

## License

The code and accompanying data are available under the [MIT License](LICENSE). External packages and model weights retain their own licenses.
