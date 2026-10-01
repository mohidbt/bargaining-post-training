"""Searched history-blind offer policies with standing-offer-reactive acceptance."""
pass


import argparse
import itertools
import random
from collections import namedtuple

from . import bots
from .env import BUYER, SELLER, Env, play, sample_draw
from .vf_env import C_HI, C_LO, V_HI, V_LO, sample_opponent_params

PRICE_MIN, PRICE_MAX = 0, 150
OPPONENTS = ("boulware", "conceder", "random_threshold")


Priors = namedtuple("Priors", "e_opp anchor_base")


def make_priors(draw_kwargs=None):
    dk = draw_kwargs or {}
    c_lo, c_hi = dk.get("c_lo", C_LO), dk.get("c_hi", C_HI)
    v_lo, v_hi = dk.get("v_lo", V_LO), dk.get("v_hi", V_HI)
    return {SELLER: Priors((v_lo + v_hi) / 2, v_hi), BUYER: Priors((c_lo + c_hi) / 2, c_lo)}


DEFAULT_PRIORS = make_priors()


def _clamp(p):
    return max(PRICE_MIN, min(PRICE_MAX, round(p)))


class ConcessionBlind:


    def __init__(self, role, own_value, e, anchor_off, margin, floor_frac, priors=None):
        self.role = role
        self.e = e
        self.margin = margin
        self.sign = 1 if role == SELLER else -1
        prior = (priors or DEFAULT_PRIORS)[role]
        m_hat = (own_value + prior.e_opp) / 2
        self.floor = own_value + floor_frac * (m_hat - own_value)
        base = prior.anchor_base
        self.anchor = _clamp(base + anchor_off if role == SELLER else base - anchor_off)

    def _better(self, a, b):
        return a >= b if self.role == SELLER else a <= b

    def target(self, env):
        frac = bots._time(env) ** (1 / self.e)
        t = round(self.anchor + (self.floor - self.anchor) * frac)
        return t if self._better(t, self.floor) else round(self.floor)

    def act(self, env):
        target = self.target(env)
        offer = env.last_offer[1] if env.last_offer else None
        mode, x = env.draw.horizon
        if offer is not None:
            if self._better(offer, target + self.sign * self.margin):
                return "#### ACCEPT"
            if mode == "T" and env.round == x:
                return "#### ACCEPT" if self._better(offer, self.floor) else "#### WALK"
        return f"#### OFFER({target})"


class Camper(bots.MidpointCamper):


    def __init__(self, role, own_value, price=None, offset=None):
        if price is None:
            price = own_value + offset if role == SELLER else own_value - offset
        super().__init__(role, _clamp(price))

    @property
    def price(self):
        return self.m_hat


Candidate = namedtuple("Candidate", "label factory")

E_GRID = (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 4.0)
ANCHOR_OFF_GRID = (-40, -30, -20, -10, 0, 10, 20, 30, 50)
MARGIN_GRID = (-5, 0, 5, 10)
FLOOR_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PRICE_GRID = tuple(range(40, 111, 5))
OFFSET_GRID = tuple(range(0, 41, 5))


def make_candidates(draw_kwargs=None):
    priors = make_priors(draw_kwargs)
    cands = []
    for e, off, m, f in itertools.product(E_GRID, ANCHOR_OFF_GRID, MARGIN_GRID, FLOOR_GRID):
        cands.append(
            Candidate(
                f"concede(e={e},anchor_off={off},margin={m},floor={f})",
                lambda role, own, e=e, off=off, m=m, f=f: ConcessionBlind(role, own, e, off, m, f, priors),
            )
        )
    for p in PRICE_GRID:
        cands.append(Candidate(f"camp(p={p})", lambda role, own, p=p: Camper(role, own, price=p)))
    for k in OFFSET_GRID:
        cands.append(Candidate(f"camp(own+/-{k})", lambda role, own, k=k: Camper(role, own, offset=k)))
    return cands


BASELINES = {
    "oracle": lambda role, own: bots.OracleBot(role),
    "boulware(default)": lambda role, own: bots.boulware(role, own, anchor=100 if role == SELLER else 40),
    "conceder(default)": lambda role, own: bots.conceder(role, own, anchor=100 if role == SELLER else 40),
    "titfortat(default)": lambda role, own: bots.TitForTat(role, own, anchor=100 if role == SELLER else 40),
    "midpoint_camper(68)": lambda role, own: bots.MidpointCamper(role, 68),
}


Episode = namedtuple("Episode", "draw opp_name opp_e opp_anchor env_seed")


WIDE_DRAW_KWARGS = dict(c_lo=40, c_hi=60, v_lo=90, v_hi=110, t_range=(2, 6))


def make_episodes(n, seat, seed, mode=None, draw_kwargs=None):


    rng = random.Random(seed)
    dk = draw_kwargs or {}
    opp_role = BUYER if seat == SELLER else SELLER
    out = []
    while len(out) < n:
        draw = sample_draw(rng, **dk)
        name = OPPONENTS[rng.randrange(len(OPPONENTS))]
        opp_own = draw.c if opp_role == SELLER else draw.v
        e, anchor = sample_opponent_params(
            rng, name, opp_role, PRICE_MIN, PRICE_MAX, opp_own,
            c_lo=dk.get("c_lo", C_LO), v_hi=dk.get("v_hi", V_HI),
        )
        if mode is None or draw.horizon[0] == mode:
            out.append(Episode(draw, name, e, anchor, rng.getrandbits(32)))
    return out


def play_episode(factory, seat, ep):
    draw = ep.draw
    me = factory(seat, draw.c if seat == SELLER else draw.v)
    opp_role = BUYER if seat == SELLER else SELLER
    opp_own = draw.c if opp_role == SELLER else draw.v
    opp = getattr(bots, ep.opp_name)(opp_role, opp_own, anchor=ep.opp_anchor, e=ep.opp_e)
    env = Env(draw, random.Random(ep.env_seed), first_mover=seat)
    r_s, r_b = play(env, me if seat == SELLER else opp, opp if seat == SELLER else me)
    assert r_s + r_b == 0, "zero-sum invariant violated"
    r_me = r_s if seat == SELLER else r_b
    return r_me, -r_me, env.outcome


def evaluate(factory, seat, episodes):
    total = deals = 0
    for ep in episodes:
        r, _, outcome = play_episode(factory, seat, ep)
        total += r
        deals += outcome[0] == "deal"
    return total / len(episodes), deals / len(episodes)


def _score(factory, episode_sets):
    per_seat = {seat: evaluate(factory, seat, eps) for seat, eps in episode_sets.items()}
    mean = sum(m for m, _ in per_seat.values()) / len(per_seat)
    return {"mean": mean, "seller": per_seat[SELLER], "buyer": per_seat[BUYER]}


def search(n_screen=400, n_refine=4000, seed=0, top_k=20, candidates=None, mode=None,
           draw_kwargs=None):


    cands = make_candidates(draw_kwargs) if candidates is None else candidates
    screen = {s: make_episodes(n_screen, s, seed + i, mode, draw_kwargs)
              for i, s in enumerate((SELLER, BUYER))}
    ranked = sorted(cands, key=lambda c: -_score(c.factory, screen)["mean"])
    refine = {s: make_episodes(n_refine, s, seed + 100 + i, mode, draw_kwargs)
              for i, s in enumerate((SELLER, BUYER))}
    blind = [{"label": c.label, **_score(c.factory, refine)} for c in ranked[:top_k]]
    blind.sort(key=lambda r: -r["mean"])
    baselines = {name: _score(f, refine) for name, f in BASELINES.items()}
    return {
        "blind": blind,
        "baselines": baselines,
        "n_screen": n_screen,
        "n_refine": n_refine,
        "seed": seed,
        "n_candidates": len(cands),
    }


def _fmt(row):
    (sm, sd), (bm, bd) = row["seller"], row["buyer"]
    return f"{row['mean']:>8.4f} | S {sm:>7.4f} (deal {sd:.2f}) | B {bm:>7.4f} (deal {bd:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=int, default=400)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--mode", choices=("T", "geom"), default=None,
                    help="condition on horizon regime (public info a blind meta may use)")
    ap.add_argument("--draw", choices=("default", "wide"), default="default",
                    help="wide = the stage-B curriculum distribution")
    args = ap.parse_args()

    dk = WIDE_DRAW_KWARGS if args.draw == "wide" else None
    res = search(n_screen=args.screen, n_refine=args.episodes, seed=args.seed,
                 top_k=args.top, mode=args.mode, draw_kwargs=dk)
    print(f"{res['n_candidates']} blind candidates | screen n={res['n_screen']}/seat | "
          f"refine n={res['n_refine']}/seat | seed={res['seed']} | mode={args.mode or 'all'} | "
          f"draw={args.draw}")
    print("\n-- top blind candidates (refined) --")
    for row in res["blind"]:
        print(f"{_fmt(row)}  {row['label']}")
    print("\n-- baselines (same refined episode sets) --")
    for name, row in res["baselines"].items():
        print(f"{_fmt(row)}  {name}")
    ceiling, oracle = res["blind"][0], res["baselines"]["oracle"]
    print(f"\nsearched comparator = {ceiling['mean']:.4f}  ({ceiling['label']})")
    print(f"oracle        = {oracle['mean']:.4f}")
    print(f"difference    = {oracle['mean'] - ceiling['mean']:.4f}")


if __name__ == "__main__":
    main()
