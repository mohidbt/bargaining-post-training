

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paired_vs_blind import bootstrap_ci, episode, summarize
from market import bots
from market.best_response import BestResponse, expected_value
from market.env import BUYER, SELLER


def upper(seat, ep):
    opp_role = BUYER if seat == SELLER else SELLER
    opp_own = ep.draw.c if opp_role == SELLER else ep.draw.v
    bot = getattr(bots, ep.opp_name)(opp_role, opp_own, anchor=ep.opp_anchor, e=ep.opp_e)
    br = BestResponse(ep.draw, seat, bot)
    ev, outcome = expected_value(br, seat, ep.draw, bot)
    assert abs(ev - br.value) < 1e-9, (ev, br.value)
    return ev, outcome


def paired(episodes_dir, seat, seed, n_boot):
    rows = [r for r in json.loads((episodes_dir / f"episodes-{seat}.json").read_text())
            if r["status"] == "ok"]
    plays = [upper(seat, episode(r)) for r in rows]
    model = [r["reward"] for r in rows]
    ref = [p[0] for p in plays]
    model_share = [(m + 1) / 2 if r["metrics"]["deal_rate"] == 1 else 0.0
                   for m, r in zip(model, rows)]
    ref_share = [(u + 1) / 2 if p[1] and p[1][0] == "deal" else 0.0 for u, p in zip(ref, plays)]
    return {
        "n": len(rows),
        "midpoint_reward": summarize(model, ref, seed, n_boot),
        "surplus_share": summarize(model_share, ref_share, seed, n_boot),
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
        "reference": "full-information best response (market.best_response.BestResponse)",
        "bootstrap": {"seed": args.boot_seed, "B": args.boot},
        "results": {seat: paired(episodes_dir, seat, args.boot_seed, args.boot)
                    for seat in (SELLER, BUYER)},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
