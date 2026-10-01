"""Exact best response with access to opponent parameters."""
pass


import argparse
import random
import statistics
from types import SimpleNamespace

from . import bots
from .env import BUYER, SELLER, Env, _deal_rewards, parse_action, sample_draw
from .vf_env import C_LO, V_HI, sample_opponent_params

PRICE_MIN, PRICE_MAX = 0, 150


class BestResponse:

    def __init__(self, draw, seat, bot, price_min=PRICE_MIN, price_max=PRICE_MAX):
        if isinstance(bot, bots.TitForTat):
            raise ValueError("TitForTat is stateful (my_last/opp_prev); out of DP scope")
        if not isinstance(bot, (bots.TimeBot, bots.RandomThreshold)):
            raise ValueError(f"unsupported opponent: {type(bot).__name__}")
        self.draw, self.seat, self.bot = draw, seat, bot
        self.price_min, self.price_max = price_min, price_max
        self.mode, self.x = draw.horizon
        if self.mode == "geom":
            total = max(2, round(1 / (1 - self.x)))
            t = total - 1
            self._stat = t if t % 2 == 1 else t + 1
        self._memo, self._stack = {}, set()
        self.value = self._V(1, None)

    def _deal(self, p):
        r_s, r_b = _deal_rewards(self.draw.c, self.draw.v, p)
        return r_s if self.seat == SELLER else r_b

    def _opp(self, rnd, p):

        fake = SimpleNamespace(draw=self.draw, round=rnd, last_offer=(self.seat, p))
        return parse_action(self.bot.act(fake), fake.last_offer, self.price_min, self.price_max)

    def _round(self, r):
        assert r % 2 == 1, "learner moves first; learner turns are odd rounds"
        return r if self.mode == "T" else min(r, self._stat)

    def _V(self, r, o):
        key = (self._round(r), o)
        if key in self._memo:
            return self._memo[key]
        if key in self._stack:
            return None
        self._stack.add(key)
        v = max(val for val, _, _ in self._actions(*key))
        self._stack.discard(key)
        self._memo[key] = v
        return v

    def _actions(self, r, o):


        acts = []
        if o is not None:
            acts.append((self._deal(o), 0, "#### ACCEPT"))
        acts.append((0.0, 1, "#### WALK"))
        if self.mode == "T" and r >= self.x:
            return acts
        delta = self.x if self.mode == "geom" else 1.0
        best = None
        rejected = {}
        for p in range(self.price_min, self.price_max + 1):
            resp = self._opp(r + 1, p)
            if resp == ("ACCEPT",):
                val = delta * self._deal(p)
            elif resp is None:


                val = delta * 1.0
            else:
                rejected.setdefault(resp, p)
                continue
            if best is None or val > best[0]:
                best = (val, 2, f"#### OFFER({p})")
        if best is not None:
            acts.append(best)

        assert sum(k[0] == "OFFER" for k in rejected) <= 1
        for resp, p in rejected.items():
            if resp == ("WALK",):
                acts.append((0.0, 3, f"#### OFFER({p})"))
                continue
            q = resp[1]
            if self.mode == "T":
                cont = 0.0 if r + 1 >= self.x else self._V(r + 2, q)
            else:
                cont = self._V(r + 2, q)
                cont = None if cont is None else delta * delta * cont
            if cont is not None:
                acts.append((cont, 3, f"#### OFFER({p})"))
        return acts

    def act(self, env):
        o = env.last_offer[1] if env.last_offer else None
        acts = self._actions(self._round(env.round), o)
        return max(acts, key=lambda a: (a[0], -a[1]))[2]


_NO_BREAK = SimpleNamespace(random=lambda: 0.0)


def expected_value(policy, seat, draw, bot):


    opp_role = BUYER if seat == SELLER else SELLER
    actors = {seat: policy, opp_role: bot}
    mode, x = draw.horizon
    total = x if mode == "T" else max(2, round(1 / (1 - x)))
    env = Env(draw, _NO_BREAK, first_mover=seat)
    while not env.done and env.round <= 2 * total + 8:
        env.step(actors[env.turn].act(env))
    if not env.done:
        return 0.0, None
    r_s, r_b = env.rewards()
    r_line = r_s if seat == SELLER else r_b
    ev = x ** (env.round - 1) * r_line if mode == "geom" else r_line
    return ev, env.outcome


def _cell(seat, opp_name, mode, n, seed, draw_kwargs=None):


    rng = random.Random(seed)
    dk = draw_kwargs or {}
    opp_role = BUYER if seat == SELLER else SELLER
    pairs = []
    while len(pairs) < n:
        draw = sample_draw(rng, **dk)
        if draw.horizon[0] != mode:
            continue
        opp_own = draw.c if opp_role == SELLER else draw.v
        e, anchor = sample_opponent_params(
            rng, opp_name, opp_role, PRICE_MIN, PRICE_MAX, opp_own,
            c_lo=dk.get("c_lo", C_LO), v_hi=dk.get("v_hi", V_HI),
        )
        bot = getattr(bots, opp_name)(opp_role, opp_own, anchor=anchor, e=e)
        dp = BestResponse(draw, seat, bot).value
        oracle, _ = expected_value(bots.OracleBot(seat), seat, draw, bot)
        pairs.append((dp, oracle))
    return pairs


def _mean_se(xs):
    return statistics.fmean(xs), statistics.stdev(xs) / len(xs) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="episodes per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draw", choices=("default", "wide"), default="default",
                    help="wide = the stage-B curriculum distribution (T-only)")
    args = ap.parse_args()

    from .blind_search import WIDE_DRAW_KWARGS
    dk = WIDE_DRAW_KWARGS if args.draw == "wide" else None
    modes = ("T",) if args.draw == "wide" else ("T", "geom")
    print(f"DP best response vs revealed-param scripted pool | n={args.n}/cell | "
          f"seed={args.seed} | draw={args.draw}")
    print(f"{'opponent':<17} {'mode':<5} {'seat':<7} {'DP mean':>16} {'oracle mean':>16} {'DP-oracle':>16}")
    by_mode = {}
    for mode in modes:
        for i, opp_name in enumerate(("boulware", "conceder", "random_threshold")):
            for j, seat in enumerate((SELLER, BUYER)):
                pairs = _cell(seat, opp_name, mode, args.n, args.seed + 10 * i + j, dk)
                dp, dse = _mean_se([d for d, _ in pairs])
                orc, ose = _mean_se([o for _, o in pairs])
                gap, gse = _mean_se([d - o for d, o in pairs])
                by_mode.setdefault(mode, []).append(pairs)
                print(f"{opp_name:<17} {mode:<5} {seat:<7} {dp:>9.4f}±{dse:.4f} "
                      f"{orc:>9.4f}±{ose:.4f} {gap:>9.4f}±{gse:.4f}")
    print("\n-- per-mode aggregates (uniform over 3 opponents, both seats) --")

    ceiling = {"T": 0.1625, "geom": 0.1279} if args.draw == "default" else {}
    for mode, cells in by_mode.items():
        pairs = [p for cell in cells for p in cell]
        dp, _ = _mean_se([d for d, _ in pairs])
        orc, _ = _mean_se([o for _, o in pairs])
        gap, gse = _mean_se([d - o for d, o in pairs])
        tail = (f" | preregistered searched comparator {ceiling[mode]:.4f}"
                if mode in ceiling else " | searched comparator: run blind_search --draw wide")
        print(f"{mode:<5} DP {dp:.4f} | oracle {orc:.4f} | oracle suboptimality "
              f"{gap:.4f}±{gse:.4f}{tail}")


if __name__ == "__main__":
    main()
