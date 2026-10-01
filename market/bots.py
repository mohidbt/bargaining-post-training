"""Scripted negotiation policies."""
pass


from .env import SELLER


def _time(env):
    mode, x = env.draw.horizon
    total = x if mode == "T" else max(2, round(1 / (1 - x)))
    return min(1.0, (env.round - 1) / max(1, total - 1))


class TimeBot:


    def __init__(self, role, own_value, e, anchor):
        self.role = role
        self.reservation = own_value
        self.e = e
        self.anchor = anchor

    def _better(self, a, b):
        return a >= b if self.role == SELLER else a <= b

    def target(self, env):
        frac = _time(env) ** (1 / self.e)
        return round(self.anchor + (self.reservation - self.anchor) * frac)

    def act(self, env):
        target = self.target(env)
        offer = env.last_offer[1] if env.last_offer else None
        mode, x = env.draw.horizon
        last_chance = mode == "T" and env.round == x
        if offer is not None:
            if self._better(offer, target):
                return "#### ACCEPT"
            if last_chance:
                return "#### ACCEPT" if self._better(offer, self.reservation) else "#### WALK"
        if not self._better(target, self.reservation):
            target = self.reservation
        return f"#### OFFER({target})"


def boulware(role, own_value, anchor, e=0.2):
    return TimeBot(role, own_value, e, anchor)


def conceder(role, own_value, anchor, e=3.0):
    return TimeBot(role, own_value, e, anchor)


class RandomThreshold:


    def __init__(self, role, own_value, threshold):
        self.role = role
        clamp = max if role == SELLER else min
        self.threshold = clamp(threshold, own_value)

    def _better(self, a, b):
        return a >= b if self.role == SELLER else a <= b

    def act(self, env):
        offer = env.last_offer[1] if env.last_offer else None
        if offer is not None and self._better(offer, self.threshold):
            return "#### ACCEPT"
        return f"#### OFFER({self.threshold})"


def random_threshold(role, own_value, anchor, e=None):


    return RandomThreshold(role, own_value, anchor)


class MidpointCamper:


    def __init__(self, role, m_hat):
        self.role = role
        self.m_hat = m_hat

    def act(self, env):
        offer = env.last_offer[1] if env.last_offer else None
        if offer is not None:
            good = offer >= self.m_hat if self.role == SELLER else offer <= self.m_hat
            if good:
                return "#### ACCEPT"
        return f"#### OFFER({self.m_hat})"


class OracleBot:


    def __init__(self, role):
        self.role = role

    def _better(self, a, b):
        return a >= b if self.role == SELLER else a <= b

    def act(self, env):
        c, v = env.draw.c, env.draw.v
        m = (c + v) / 2
        opp_reservation = v if self.role == SELLER else c
        mode, x = env.draw.horizon
        if mode == "geom":

            frac, accept_floor = _time(env), m
        else:
            frac = _time(env) ** 10
            accept_floor = round(opp_reservation + (m - opp_reservation) * frac)
        target = round(opp_reservation + (m - opp_reservation) * frac)
        offer = env.last_offer[1] if env.last_offer else None
        if offer is not None:
            if self._better(offer, accept_floor):
                return "#### ACCEPT"
            if mode == "T" and env.round == x:
                return "#### ACCEPT" if self._better(offer, m) else "#### WALK"
        return f"#### OFFER({target})"


class TitForTat(TimeBot):

    def __init__(self, role, own_value, anchor):
        super().__init__(role, own_value, e=1.0, anchor=anchor)
        self.my_last = None
        self.opp_prev = None

    def target(self, env):
        opp_last = env.last_offer[1] if env.last_offer else None
        if self.my_last is None:
            target = self.anchor
        else:
            concession = abs(opp_last - self.opp_prev) if self.opp_prev is not None else 0
            step = -concession if self.role == SELLER else concession
            target = self.my_last + step
        if not self._better(target, self.reservation):
            target = self.reservation
        self.opp_prev = opp_last
        self.my_last = target
        return target
