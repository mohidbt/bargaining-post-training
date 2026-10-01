

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market import blind_search as bs
from market import bots
from market.best_response import BestResponse
from market.env import BUYER, SELLER

DK = bs.WIDE_DRAW_KWARGS
FLOOR_EXT = (1.25, 1.5, 2.0)
SEATS = (SELLER, BUYER)
OLD_WINNER = "concede(e=0.02,anchor_off=-20,margin=0,floor=1.0)"


def mean_se(xs):
    return statistics.fmean(xs), statistics.stdev(xs) / math.sqrt(len(xs))


def dp_value(seat, ep):
    opp_role = BUYER if seat == SELLER else SELLER
    opp_own = ep.draw.c if opp_role == SELLER else ep.draw.v
    bot = getattr(bots, ep.opp_name)(opp_role, opp_own, anchor=ep.opp_anchor, e=ep.opp_e)
    return BestResponse(ep.draw, seat, bot).value


def summarize(values, episodes):
    out = {"family": {}, "seat": {}}
    for fam in bs.OPPONENTS:
        out["family"][fam] = {}
        for seat in SEATS:
            xs = [v for v, ep in zip(values[seat], episodes[seat]) if ep.opp_name == fam]
            m, se = mean_se(xs)
            out["family"][fam][seat] = {"mean": m, "se": se, "n": len(xs)}
    for seat in SEATS:
        m, se = mean_se(values[seat])
        out["seat"][seat] = {"mean": m, "se": se, "n": len(values[seat])}
    out["pooled"] = statistics.fmean(out["seat"][s]["mean"] for s in SEATS)
    out["pooled_se"] = math.sqrt(sum(out["seat"][s]["se"] ** 2 for s in SEATS)) / 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bs.FLOOR_GRID = bs.FLOOR_GRID + FLOOR_EXT
    res = bs.search(n_screen=400, n_refine=8000, seed=0, top_k=20, draw_kwargs=DK)
    lines = [f"{res['n_candidates']} candidates | screen 400/seat | refine 8000/seat | seed 0 | "
             f"draw wide | FLOOR_GRID={bs.FLOOR_GRID}", "", "-- top candidates (refined) --"]
    lines += [f"{bs._fmt(r)}  {r['label']}" for r in res["blind"]]
    lines += ["", "-- baselines --"]
    lines += [f"{bs._fmt(r)}  {name}" for name, r in res["baselines"].items()]

    by_label = {c.label: c for c in bs.make_candidates(DK)}
    winner = res["blind"][0]["label"]
    episodes = {s: bs.make_episodes(8000, s, 100 + i, None, DK) for i, s in enumerate(SEATS)}
    play = lambda label, seat: [bs.play_episode(by_label[label].factory, seat, ep)[0]
                                for ep in episodes[seat]]
    new = {s: play(winner, s) for s in SEATS}
    old = {s: play(OLD_WINNER, s) for s in SEATS}
    dp = {s: [dp_value(s, ep) for ep in episodes[s]] for s in SEATS}

    result = {
        "episodes": "blind_search.make_episodes(8000, seat, 100+i, None, WIDE_DRAW_KWARGS)",
        "n_candidates": res["n_candidates"],
        "winner": winner,
        "upper_full_information": summarize(dp, episodes),
        "open_loop_winner": summarize(new, episodes),
        "previous_winner": {"label": OLD_WINNER, **summarize(old, episodes)},
        "winner_minus_previous": summarize(
            {s: [a - b for a, b in zip(new[s], old[s])] for s in SEATS}, episodes),
    }
    lines += ["", f"winner: {winner}"]
    for key in ("upper_full_information", "open_loop_winner", "previous_winner",
                "winner_minus_previous"):
        r = result[key]
        lines += ["", f"== {key}: pooled {r['pooled']:.4f} (SE {r['pooled_se']:.4f})"]
        for seat in SEATS:
            c = r["seat"][seat]
            lines.append(f"  {seat:<7} {c['mean']:.4f} (SE {c['se']:.4f}, n={c['n']})")
        for fam in bs.OPPONENTS:
            cells = "  ".join(f"{seat} {r['family'][fam][seat]['mean']:.3f}±"
                              f"{r['family'][fam][seat]['se']:.3f} (n={r['family'][fam][seat]['n']})"
                              for seat in SEATS)
            lines.append(f"  {fam:<17} {cells}")
    text = "\n".join(lines)
    print(text)
    (out_dir / "bounds.out").write_text(text + "\n")
    (out_dir / "bounds.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
