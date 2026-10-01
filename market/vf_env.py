"""Prompts, seeded datasets and the Verifiers environment adapter."""
pass


import math
import random

import verifiers as vf
from datasets import Dataset

from . import bots
from .env import BUYER, SELLER, Draw, Env, sample_draw

C_LO, C_HI, V_LO, V_HI = 40, 80, 50, 100


def priors_line(c_lo, c_hi, v_lo, v_hi):


    p_no_zopa = sum(
        v <= c for c in range(c_lo, c_hi + 1) for v in range(v_lo, v_hi + 1)
    ) / ((c_hi - c_lo + 1) * (v_hi - v_lo + 1))
    return (
        f"PRIORS (public): c ~ U[{c_lo},{c_hi}], v ~ U[{v_lo},{v_hi}], rho = 0. "
        f"P(no-ZOPA) = {p_no_zopa:.2f}."
    )


PRIORS = priors_line(C_LO, C_HI, V_LO, V_HI)

BOT_FACTORIES = {
    "boulware": bots.boulware,
    "conceder": bots.conceder,
    "random_threshold": bots.random_threshold,
}


RETRY_MESSAGE = (
    "Invalid action, one retry left before forfeit. "
    'End your reply with exactly "#### OFFER(N)" (N your integer '
    'price), "#### ACCEPT", or "#### WALK".'
)


RETRY_MESSAGE_OPENING = (
    "Invalid action, one retry left before forfeit. There is no standing "
    'offer, so ACCEPT is not available. End your reply with exactly '
    '"#### OFFER(N)" (N your integer price) or "#### WALK".'
)


def render_prompt(
    draw, seat, price_min, price_max, scoring="negated", termination="standard", priors=PRIORS
):


    if seat == SELLER:
        role = f"ROLE: SELLER. Your cost c = {draw.c} (private)."
        side = "above"
        own_formula = "(2p - v - c)/(v - c)"
    else:
        role = f"ROLE: BUYER. Your value v = {draw.v} (private)."
        side = "below"
        own_formula = "(v + c - 2p)/(v - c)"
    mode, x = draw.horizon
    if mode == "T":
        horizon = f"revealed, T = {x} rounds. Current round in each message."
    else:
        horizon = f"geometric, continuation prob delta = {x:.2f}. Current round in each message."
    if scoring == "negated":
        score_head = (
            "SCORING (public): a deal at price p scores (2p - v - c)/(v - c), clamped to "
            "[-1, 1], for the seller and the negative for the buyer."
        )
    elif scoring == "rolelocal":
        score_head = (
            f"SCORING (public): a deal at price p scores {own_formula}, clamped to "
            f"[-1, 1], for you. The game is zero-sum."
        )
    else:
        raise ValueError(f"unknown scoring: {scoring!r}")
    if termination == "standard":
        term = ""
    elif termination == "reserve":
        term = (
            " Think briefly and leave room to finish: a reply that runs out of "
            "space before its final action line is invalid."
        )
    else:
        raise ValueError(f"unknown termination: {termination!r}")
    return (
        f"{role}\n"
        f"{priors}\n"
        f"HORIZON: {horizon}\n"
        f"RULES: alternate turns, integer prices {price_min}-{price_max}. ACCEPT binds at "
        f"opponent's last offer. WALK ends at (0,0). Invalid output: 1 retry, then forfeit.\n"
        f"{score_head} Your score is positive "
        f"only if p is {side} the midpoint of c and v. If v <= c (no ZOPA), a deal scores "
        f"+1/-1 by the same midpoint rule (0 exactly at it). WALK, deadline, and "
        f"breakdown score 0 for both. Forfeiting by invalid output scores you -1, the "
        f"opponent +1. A deal can score worse than walking away.\n"
        f'Your reply MUST end with one line that is exactly "#### OFFER(N)" '
        f'(N replaced by your integer price), "#### ACCEPT", or "#### WALK". '
        f"Nothing may follow that line. "
        f"Replies without such a final line waste your retry, then forfeit."
        f"{term}"
    )


E_RANGES = {"boulware": (0.1, 0.5), "conceder": (2.0, 4.0)}


def sample_opponent_params(rng, name, role, price_min, price_max, own_value=None,
                           c_lo=C_LO, v_hi=V_HI):


    if name == "random_threshold":
        if role == SELLER:
            return None, rng.randint(own_value, v_hi)
        return None, rng.randint(c_lo, own_value)
    lo, hi = E_RANGES[name]
    e = rng.uniform(lo, hi)
    if role == SELLER:
        anchor = rng.randint(v_hi - 10, min(price_max, v_hi + 30))
    else:
        anchor = rng.randint(max(price_min, c_lo - 30), c_lo + 10)
    return e, anchor


def _make_opponent(draw, role, name, price_min, price_max, e=None, anchor=None):
    own_value = draw.c if role == SELLER else draw.v
    if anchor is None:
        anchor = price_max if role == SELLER else price_min
    if e is None:
        return BOT_FACTORIES[name](role, own_value, anchor)
    return BOT_FACTORIES[name](role, own_value, anchor, e=e)


def _as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "") for b in content)
        return " ".join(p for p in parts if p)
    return ""


def _last_assistant_text(messages):
    for m in reversed(messages):
        role = m.role if hasattr(m, "role") else m["role"]
        if role == "assistant":
            return _as_text(m.content if hasattr(m, "content") else m["content"])
    return ""


def _outcome_line(outcome):
    kind = outcome[0]
    if kind == "deal":
        return f"Deal at {outcome[1]}."
    if kind == "walk":
        return "Walkaway. No deal."
    if kind == "forfeit":
        return "Forfeit."
    if kind == "timeout":
        return "Deadline reached. No deal."
    return "Breakdown. No deal."


def bargaining_reward(state, **kwargs):
    market = state.get("market")
    if not market or not market["env"].done:
        return 0.0
    r_seller, r_buyer = market["env"].rewards()
    return r_seller if market["seat"] == SELLER else r_buyer


def _outcome_flag(state, kind):
    market = state.get("market")
    if not market or not market["env"].done:
        return 0.0
    return 1.0 if market["env"].outcome[0] == kind else 0.0


def deal_rate(state, **kwargs):
    return _outcome_flag(state, "deal")


def positive_deal_rate(state, **kwargs):
    return _outcome_flag(state, "deal") if bargaining_reward(state) > 0 else 0.0


def negative_deal_rate(state, **kwargs):
    return _outcome_flag(state, "deal") if bargaining_reward(state) < 0 else 0.0


def walk_rate(state, **kwargs):
    return _outcome_flag(state, "walk")


def forfeit_rate(state, **kwargs):
    return _outcome_flag(state, "forfeit")


def timeout_rate(state, **kwargs):
    return _outcome_flag(state, "timeout")


def breakdown_rate(state, **kwargs):
    return _outcome_flag(state, "breakdown")


def zopa_rate(state, **kwargs):
    info = state.get("info") or {}
    return 1.0 if info.get("v", 0) > info.get("c", 0) else 0.0


def zopa_walk_rate(state, **kwargs):

    return walk_rate(state) * zopa_rate(state)


def learner_walk_rate(state, **kwargs):
    market = state.get("market")
    if not market or not market["env"].done:
        return 0.0
    outcome = market["env"].outcome
    info = state.get("info") or {}
    return 1.0 if outcome[0] == "walk" and outcome[1] == info.get("seat") else 0.0


def zopa_learner_walk_rate(state, **kwargs):
    return learner_walk_rate(state) * zopa_rate(state)


_METRIC_FUNCS = [
    deal_rate,
    positive_deal_rate,
    negative_deal_rate,
    walk_rate,
    forfeit_rate,
    timeout_rate,
    breakdown_rate,
    zopa_rate,
    zopa_walk_rate,
    learner_walk_rate,
    zopa_learner_walk_rate,
]


class BargainingEnv(vf.MultiTurnEnv):
    def __init__(self, dataset, max_turns=60, retry_style="legacy", **kwargs):
        if retry_style not in ("legacy", "opening_aware"):
            raise ValueError(f"unknown retry_style: {retry_style!r}")
        rubric = vf.Rubric(
            funcs=[bargaining_reward, *_METRIC_FUNCS],
            weights=[1.0] + [0.0] * len(_METRIC_FUNCS),
        )
        super().__init__(dataset=dataset, rubric=rubric, max_turns=max_turns, **kwargs)
        self.retry_style = retry_style

    async def setup_state(self, state):
        info = state["info"]
        h = info["horizon"]
        mode = h["mode"]
        param = int(h["param"]) if mode == "T" else float(h["param"])
        draw = Draw(c=info["c"], v=info["v"], horizon=(mode, param))
        seat = info["seat"]
        env = Env(
            draw,
            random.Random(info["seed"]),
            first_mover=seat,
            price_min=info["price_min"],
            price_max=info["price_max"],
        )
        opp_role = BUYER if seat == SELLER else SELLER
        opponent = _make_opponent(
            draw,
            opp_role,
            info["opponent"],
            info["price_min"],
            info["price_max"],
            e=info.get("opp_e"),
            anchor=info.get("opp_anchor"),
        )
        state["market"] = {"env": env, "opponent": opponent, "seat": seat, "responses": 0}

    def _finalize(self, state, env, opp_text=None):
        parts = [] if opp_text is None else [f"Opponent: {opp_text}"]
        parts.append(_outcome_line(env.outcome))
        response = [vf.UserMessage(content=" ".join(parts))]
        state["final_env_response"] = response
        return response

    async def env_response(self, messages, state, **kwargs):
        market = state["market"]
        env, opponent, seat = market["env"], market["opponent"], market["seat"]

        market["responses"] += 1
        env.step(_last_assistant_text(messages))
        if env.done:
            return self._finalize(state, env)
        if env.turn == seat:
            if self.retry_style == "opening_aware" and env.last_offer is None:
                return [vf.UserMessage(content=RETRY_MESSAGE_OPENING)]
            return [vf.UserMessage(content=RETRY_MESSAGE)]

        opp_text = opponent.act(env)
        env.step(opp_text)
        if env.done:
            return self._finalize(state, env, opp_text)
        return [vf.UserMessage(content=f"Round {env.round}. Opponent: {opp_text}")]

    @vf.cleanup
    async def flush_pending_action(self, state):


        market = state.get("market")
        if not market or market["env"].done:
            return
        trajectory = state.get("trajectory") or []
        if len(trajectory) > market["responses"]:
            env = market["env"]
            env.step(_last_assistant_text(trajectory[-1]["completion"]))
            if not env.done and env.turn != market["seat"]:
                env.step(market["opponent"].act(env))


def validate_draw_bounds(price_min, price_max, c_lo, c_hi, v_lo, v_hi, t_lo, t_hi):


    if (t_lo is None) != (t_hi is None):
        raise ValueError("t_lo and t_hi must be given together")
    if not (c_lo <= c_hi and v_lo <= v_hi):
        raise ValueError(f"bad draw bounds: c[{c_lo},{c_hi}] v[{v_lo},{v_hi}]")


    if not (c_lo <= v_lo and c_hi <= v_hi):
        raise ValueError(
            f"supports must satisfy c_lo <= v_lo and c_hi <= v_hi: "
            f"c[{c_lo},{c_hi}] v[{v_lo},{v_hi}]")
    if not (price_min <= c_lo and v_hi <= price_max):
        raise ValueError(
            f"supports must lie inside the price grid [{price_min},{price_max}]: "
            f"c_lo={c_lo}, v_hi={v_hi}")
    if t_lo is not None and not (2 <= t_lo <= t_hi):
        raise ValueError(f"bad t range: [{t_lo},{t_hi}] (need 2 <= t_lo <= t_hi)")
    return None if t_lo is None else (t_lo, t_hi)


def make_dataset(
    n, seed=0, opponent="boulware", seat=SELLER, price_min=0, price_max=150, mix_weights=None,
    scoring="negated", termination="standard",
    c_lo=C_LO, c_hi=C_HI, v_lo=V_LO, v_hi=V_HI, t_lo=None, t_hi=None,
):


    t_range = validate_draw_bounds(price_min, price_max, c_lo, c_hi, v_lo, v_hi, t_lo, t_hi)
    priors = priors_line(c_lo, c_hi, v_lo, v_hi)
    if mix_weights is not None:
        if opponent != "mix":
            raise ValueError("mix_weights requires opponent='mix'")
        unknown = set(mix_weights) - set(BOT_FACTORIES)
        if unknown:
            raise ValueError(f"unknown opponents in mix_weights: {sorted(unknown)}")
        if any(not (w >= 0 and math.isfinite(w)) for w in mix_weights.values()):
            raise ValueError(f"mix_weights must be finite and non-negative: {mix_weights}")
        if sum(mix_weights.values()) <= 0:
            raise ValueError(f"mix_weights must have positive sum: {mix_weights}")
    names = sorted(BOT_FACTORIES)
    weights = [mix_weights.get(k, 0.0) for k in names] if mix_weights else None
    assignment = None
    if opponent == "mix" and weights is None:
        assignment = [names[j % len(names)] for j in range(n)]
        random.Random((seed << 34) | 3).shuffle(assignment)
    rows = []
    opp_role = BUYER if seat == SELLER else SELLER
    for i in range(n):
        game_rng = random.Random((seed << 34) | (i << 2))
        opp_rng = random.Random((seed << 34) | (i << 2) | 1)
        draw = sample_draw(game_rng, c_lo, c_hi, v_lo, v_hi, t_range)
        mode, param = draw.horizon
        env_seed = game_rng.getrandbits(32)
        opp_own = draw.c if opp_role == SELLER else draw.v
        if opponent != "mix":
            opp_name = opponent
        elif weights is None:
            opp_name = assignment[i]
        else:
            opp_name = opp_rng.choices(names, weights=weights)[0]
        opp_e, opp_anchor = sample_opponent_params(
            opp_rng, opp_name, opp_role, price_min, price_max, opp_own,
            c_lo=c_lo, v_hi=v_hi,
        )
        info = {
            "c": draw.c,
            "v": draw.v,
            "horizon": {"mode": mode, "param": float(param)},
            "seat": seat,
            "opponent": opp_name,
            "opp_e": opp_e,
            "opp_anchor": opp_anchor,
            "seed": env_seed,
            "price_min": price_min,
            "price_max": price_max,
        }


        prompt = [
            {
                "role": "system",
                "content": render_prompt(
                    draw, seat, price_min, price_max,
                    scoring=scoring, termination=termination, priors=priors,
                ),
            },
            {"role": "user", "content": "Round 1. You open."},
        ]
        rows.append({"prompt": prompt, "answer": "", "info": info})
    return Dataset.from_list(rows)


def load_environment(
    n=64, seed=0, task="bargain", opponent="boulware", seat=SELLER, mix_weights=None,
    scoring="negated", termination="standard",
    c_lo=C_LO, c_hi=C_HI, v_lo=V_LO, v_hi=V_HI, t_lo=None, t_hi=None, **kwargs
):


    if task != "bargain":
        if opponent != "boulware" or mix_weights is not None:
            raise ValueError(f"probe task {task!r} takes no opponent/mix_weights")
        from .vf_probe import load_probe


        return load_probe(
            task, n=n, seed=seed, seat=seat, scoring=scoring, termination=termination,
            c_lo=c_lo, c_hi=c_hi, v_lo=v_lo, v_hi=v_hi, t_lo=t_lo, t_hi=t_hi, **kwargs
        )
    return BargainingEnv(
        dataset=make_dataset(
            n, seed, opponent, seat, mix_weights=mix_weights,
            scoring=scoring, termination=termination,
            c_lo=c_lo, c_hi=c_hi, v_lo=v_lo, v_hi=v_hi, t_lo=t_lo, t_hi=t_hi,
        ),
        **kwargs,
    )
