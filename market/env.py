"""Bilateral bargaining mechanics and midpoint rewards."""
pass


import re
from dataclasses import dataclass

SELLER, BUYER = "seller", "buyer"


THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
ACTION_RE = re.compile(r"^\s*####\s*(OFFER\(\s*(-?\d+)\s*\)|ACCEPT|WALK)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Draw:
    c: int
    v: int
    horizon: tuple


def parse_action(text, last_offer, price_min, price_max):


    matches = ACTION_RE.finditer(THINK_RE.sub("", text))
    m = next(matches, None)
    if m is None or next(matches, None) is not None:
        return None
    if m.group(1) == "ACCEPT":
        return ("ACCEPT",) if last_offer else None
    if m.group(1) == "WALK":
        return ("WALK",)
    p = int(m.group(2))
    return ("OFFER", p) if price_min <= p <= price_max else None


def _deal_rewards(c, v, p):


    w = v - c
    d = 2 * p - v - c
    if w <= 0:
        r_s = 0.0 if d == 0 else (1.0 if d > 0 else -1.0)
    else:
        r_s = max(-1.0, min(1.0, d / w))
    return (r_s, -r_s)


def sample_draw(rng, c_lo=40, c_hi=80, v_lo=50, v_hi=100, t_range=None):


    if t_range is not None:
        horizon = ("T", rng.randint(*t_range))
    elif rng.random() < 0.5:
        horizon = ("T", rng.randint(2, 20))
    else:
        horizon = ("geom", rng.uniform(0.85, 0.99))
    return Draw(c=rng.randint(c_lo, c_hi), v=rng.randint(v_lo, v_hi), horizon=horizon)


class Env:


    def __init__(self, draw, rng, first_mover=SELLER, price_min=0, price_max=150):
        self.draw = draw
        self.rng = rng
        self.price_min = price_min
        self.price_max = price_max
        self.turn = first_mover
        self.round = 1
        self.last_offer = None
        self.transcript = []
        self.outcome = None
        self.retry_used = False

    @property
    def done(self):
        return self.outcome is not None

    def _parse(self, text):
        return parse_action(text, self.last_offer, self.price_min, self.price_max)

    def step(self, text):
        assert not self.done
        action = self._parse(text)
        if action is None:
            if self.retry_used:
                self.outcome = ("forfeit", self.turn)
            else:
                self.retry_used = True
            return self.outcome
        self.retry_used = False

        kind = action[0]
        if kind == "ACCEPT":
            self.outcome = ("deal", self.last_offer[1])
        elif kind == "WALK":
            self.outcome = ("walk", self.turn)
        else:
            self.last_offer = (self.turn, action[1])
        self.transcript.append((self.turn, text))
        if self.done:
            return self.outcome

        mode, x = self.draw.horizon
        if mode == "T" and self.round >= x:
            self.outcome = ("timeout",)
        elif mode == "geom" and self.rng.random() >= x:
            self.outcome = ("breakdown",)
        else:
            self.round += 1
            self.turn = BUYER if self.turn == SELLER else SELLER
        return self.outcome

    def rewards(self):
        assert self.done
        kind = self.outcome[0]
        if kind == "forfeit":
            r_s = -1.0 if self.outcome[1] == SELLER else 1.0
            return (r_s, -r_s)
        if kind != "deal":
            return (0.0, 0.0)
        return _deal_rewards(self.draw.c, self.draw.v, self.outcome[1])


def play(env, seller_bot, buyer_bot):
    bots = {SELLER: seller_bot, BUYER: buyer_bot}
    while not env.done:
        env.step(bots[env.turn].act(env))
    return env.rewards()
