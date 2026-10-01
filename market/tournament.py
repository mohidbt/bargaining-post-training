"""Deterministic policy comparison."""
pass


import argparse
import itertools
import random
from collections import defaultdict

from .bots import TitForTat, boulware, conceder
from .env import BUYER, SELLER, Env, play, sample_draw


BOT_FACTORIES = {
    "boulware": lambda role, own: boulware(role, own, anchor=100 if role == SELLER else 40),
    "conceder": lambda role, own: conceder(role, own, anchor=100 if role == SELLER else 40),
    "titfortat": lambda role, own: TitForTat(role, own, anchor=100 if role == SELLER else 40),
}


def run(n_games, seed):
    rng = random.Random(seed)
    reward = defaultdict(list)
    outcomes = defaultdict(int)
    zopa_deals = zopa_draws = nozopa_walks = nozopa_draws = 0

    for a, b in itertools.combinations_with_replacement(BOT_FACTORIES, 2):
        for _ in range(n_games):
            draw = sample_draw(rng)
            for s_name, b_name in ((a, b), (b, a)):
                env = Env(draw, random.Random(rng.getrandbits(32)))
                r_s, r_b = play(
                    env,
                    BOT_FACTORIES[s_name](SELLER, draw.c),
                    BOT_FACTORIES[b_name](BUYER, draw.v),
                )
                assert r_s + r_b == 0, "zero-sum invariant violated"
                reward[s_name].append(r_s)
                reward[b_name].append(r_b)
                outcomes[env.outcome[0]] += 1
                if draw.v > draw.c:
                    zopa_draws += 1
                    zopa_deals += env.outcome[0] == "deal"
                elif draw.v < draw.c:
                    nozopa_draws += 1
                    nozopa_walks += env.outcome[0] != "deal"
    return reward, outcomes, zopa_deals / zopa_draws, nozopa_walks / nozopa_draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reward, outcomes, deal_rate, walk_rate = run(args.games, args.seed)
    print(f"{'bot':<12}{'mean reward':>12}{'episodes':>10}")
    for name, rs in sorted(reward.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"{name:<12}{sum(rs) / len(rs):>12.4f}{len(rs):>10}")
    print(f"\noutcomes: {dict(outcomes)}")
    print(f"deal rate (ZOPA draws): {deal_rate:.3f}")
    print(f"no-deal rate (no-ZOPA draws): {walk_rate:.3f}")


if __name__ == "__main__":
    main()
