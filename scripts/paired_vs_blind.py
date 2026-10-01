

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market import blind_search as bs
from market.env import BUYER, SELLER, Draw


WINNERS = {
    "ext": {"params": dict(e=0.02, anchor_off=-20, margin=-5, floor_frac=1.25), "priors": "wide"},
}


def episode(row):
    i = row["info"]
    draw = Draw(i["c"], i["v"], (i["horizon"]["mode"], i["horizon"]["param"]))
    return bs.Episode(draw, i["opponent"], i["opp_e"], i["opp_anchor"], i["seed"])


def bootstrap_ci(diffs, seed, n_boot):
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return [lo, hi]


def summarize(model, blind, seed, n_boot):
    diffs = [m - b for m, b in zip(model, blind)]
    return {
        "model_mean": statistics.fmean(model),
        "blind_mean": statistics.fmean(blind),
        "paired_mean_model_minus_blind": statistics.fmean(diffs),
        "ci95": bootstrap_ci(diffs, seed, n_boot),
        "model_wins": sum(d > 0 for d in diffs),
        "blind_wins": sum(d < 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
    }


def paired(episodes_dir, seat, winner, seed, n_boot):
    params = winner["params"]
    priors = bs.make_priors(bs.WIDE_DRAW_KWARGS) if winner["priors"] == "wide" else None
    rows = [r for r in json.loads((episodes_dir / f"episodes-{seat}.json").read_text())
            if r["status"] == "ok"]
    factory = lambda role, own: bs.ConcessionBlind(role, own, priors=priors, **params)
    plays = [bs.play_episode(factory, seat, episode(r)) for r in rows]

    model = [r["reward"] for r in rows]
    blind = [p[0] for p in plays]
    model_share = [(m + 1) / 2 if r["metrics"]["deal_rate"] == 1 else 0.0
                   for m, r in zip(model, rows)]
    blind_share = [(b + 1) / 2 if p[2][0] == "deal" else 0.0 for b, p in zip(blind, plays)]
    return {
        "n": len(rows),
        "midpoint_reward": summarize(model, blind, seed, n_boot),
        "surplus_share": summarize(model_share, blind_share, seed, n_boot),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir",
                    default=str(root / "data/frontier"))
    ap.add_argument("--boot-seed", type=int, default=20260802)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    episodes_dir = Path(args.episodes_dir)
    result = {
        "episodes_dir": episodes_dir.name,
        "bootstrap": {"seed": args.boot_seed, "B": args.boot},
        "wide_priors": {r: bs.make_priors(bs.WIDE_DRAW_KWARGS)[r]._asdict()
                        for r in (SELLER, BUYER)},
        "winners": WINNERS,
        "results": {
            name: {seat: paired(episodes_dir, seat, winner, args.boot_seed, args.boot)
                   for seat in (SELLER, BUYER)}
            for name, winner in WINNERS.items()
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
