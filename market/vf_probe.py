"""Opening-action and formatting evaluations."""


import random

import verifiers as vf
from datasets import Dataset

from .env import SELLER, parse_action, sample_draw
from .vf_env import (
    C_HI, C_LO, V_HI, V_LO,
    RETRY_MESSAGE, RETRY_MESSAGE_OPENING, _last_assistant_text, priors_line, render_prompt,
    validate_draw_bounds,
)


def render_format_prompt(price_min, price_max):


    return (
        f"FORMAT CHECK: this probe tests output format only; there is no game.\n"
        f"RULES: integer prices {price_min}-{price_max}. There is no standing "
        f"offer, so ACCEPT is not available. Invalid output scores 0 (no retry).\n"
        f'Your reply MUST end with one line that is exactly "#### OFFER(N)" '
        f'(N replaced by your integer price) or "#### WALK". '
        f"Nothing may follow that line."
    )


def render_fullinfo_prompt(c, v, seat, price_min, price_max, formula="negated", style="end"):


    if seat == SELLER:
        role = f"ROLE: SELLER. Your cost c = {c}."
        disclosed = f"the buyer's value v = {v} is disclosed to you"
        side = "above"
        own_formula = "(2p - v - c)/(v - c)"
        task = (
            "compute the LOWEST integer price p that scores strictly "
            "positively for you, and submit exactly that price"
        )
    else:
        role = f"ROLE: BUYER. Your value v = {v}."
        disclosed = f"the seller's cost c = {c} is disclosed to you"
        side = "below"
        own_formula = "(v + c - 2p)/(v - c)"
        task = (
            "compute the HIGHEST integer price p that scores strictly "
            "positively for you, and submit exactly that price"
        )
    if formula == "negated":
        scoring = (
            "SCORING (public): a deal at price p scores (2p - v - c)/(v - c), "
            "clamped to [-1, 1], for the seller and the negative for the buyer."
        )
    elif formula == "rolelocal":
        scoring = (
            f"SCORING (public): a deal at price p scores {own_formula}, "
            f"clamped to [-1, 1], for you. The game is zero-sum."
        )
    elif formula == "negexplicit":
        if seat == SELLER:
            raise ValueError("formula 'negexplicit' is buyer-only")
        scoring = (
            "SCORING (public): a deal at price p scores "
            "-((2p - v - c)/(v - c)), clamped to [-1, 1], for you. "
            "The game is zero-sum."
        )
    elif formula == "noformula":
        scoring = (
            "SCORING (public): deal scores are clamped to [-1, 1]. "
            "The game is zero-sum."
        )
    else:
        raise ValueError(f"unknown formula: {formula!r}")
    if style == "end":
        fmt = (
            'Your reply MUST end with one line that is exactly "#### OFFER(N)" '
            "(N replaced by your integer price). Nothing may follow that line."
        )
    elif style == "first":
        fmt = (
            'Your reply MUST be ONLY the single line "#### OFFER(N)" '
            "(N replaced by your integer price). Output it immediately — "
            "no reasoning before or after it."
        )
    else:
        raise ValueError(f"unknown style: {style!r}")
    return (
        f"{role}\n"
        f"FULL INFORMATION (probe): {disclosed}. Nothing is hidden.\n"
        f"{scoring} "
        f"Your score is positive only if p is {side} the midpoint of c and v.\n"
        f"TASK: {task}. Any other price receives probe reward 0.\n"
        f"RULES: integer prices {price_min}-{price_max}. Invalid output scores "
        f"0 (no retry).\n"
        f"{fmt}"
    )


def _parsed(completion, info):


    return parse_action(
        _last_assistant_text(completion), None, info["price_min"], info["price_max"]
    )


def format_reward(completion, info, **kwargs):
    return 1.0 if _parsed(completion, info) is not None else 0.0


def _offer(completion, info):
    action = _parsed(completion, info)
    return action[1] if action is not None and action[0] == "OFFER" else None


def valid_action_rate(completion, info, **kwargs):
    return format_reward(completion, info)


def offer_rate(completion, info, **kwargs):
    return 1.0 if _offer(completion, info) is not None else 0.0


def reservation_safe_rate(completion, info, **kwargs):
    p = _offer(completion, info)
    if p is None:
        return 0.0
    if info["seat"] == SELLER:
        return 1.0 if p >= info["c"] else 0.0
    return 1.0 if p <= info["v"] else 0.0


def _target(info):

    s = info["c"] + info["v"]
    return s // 2 + 1 if info["seat"] == SELLER else (s + 1) // 2 - 1


def correct_side_rate(completion, info, **kwargs):


    p = _offer(completion, info)
    if p is None:
        return 0.0
    d = 2 * p - info["v"] - info["c"]
    return 1.0 if (d > 0 if info["seat"] == SELLER else d < 0) else 0.0


def opp_safe_rate(completion, info, **kwargs):

    p = _offer(completion, info)
    if p is None:
        return 0.0
    if info["seat"] == SELLER:
        return 1.0 if p <= info["v"] else 0.0
    return 1.0 if p >= info["c"] else 0.0


def within_one_rate(completion, info, **kwargs):


    p = _offer(completion, info)
    return 1.0 if p is not None and abs(p - _target(info)) <= 1 else 0.0


def fullinfo_reward(completion, info, **kwargs):
    return 1.0 if _offer(completion, info) == _target(info) else 0.0


def make_format_dataset(n, price_min=0, price_max=150):

    prompt = [
        {"role": "system", "content": render_format_prompt(price_min, price_max)},

        {"role": "user", "content": "Reply now."},
    ]
    info = {"seat": SELLER, "c": 0, "v": 0, "price_min": price_min, "price_max": price_max}
    return Dataset.from_list([{"prompt": prompt, "answer": "", "info": info}] * n)


def make_fullinfo_dataset(
    n, seed=0, seat=SELLER, price_min=0, price_max=150, formula="negated", style="end"
):


    rows = []
    i = 0
    while len(rows) < n:
        draw = sample_draw(random.Random((seed << 34) | (i << 2)))
        i += 1
        if draw.v <= draw.c:
            continue
        info = {
            "c": draw.c,
            "v": draw.v,
            "seat": seat,
            "source_row": i - 1,
            "price_min": price_min,
            "price_max": price_max,
        }
        prompt = [
            {
                "role": "system",
                "content": render_fullinfo_prompt(
                    draw.c, draw.v, seat, price_min, price_max, formula, style
                ),
            },
            {"role": "user", "content": "Submit your price."},
        ]
        rows.append({"prompt": prompt, "answer": "", "info": info})
    return Dataset.from_list(rows)


def make_format_ctx_dataset(
    n, seed=0, seat=SELLER, price_min=0, price_max=150,
    scoring="negated", termination="standard",
    c_lo=C_LO, c_hi=C_HI, v_lo=V_LO, v_hi=V_HI, t_lo=None, t_hi=None,
):


    t_range = validate_draw_bounds(price_min, price_max, c_lo, c_hi, v_lo, v_hi, t_lo, t_hi)
    priors = priors_line(c_lo, c_hi, v_lo, v_hi)
    rows = []
    for i in range(n):
        draw = sample_draw(random.Random((seed << 34) | (i << 2)), c_lo, c_hi, v_lo, v_hi, t_range)
        info = {
            "c": draw.c,
            "v": draw.v,
            "seat": seat,
            "source_row": i,
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


_INVALID_KINDS = ("prose_no_action", "missing_sentinel", "two_action_lines")


def _invalid_attempt(kind, seat, own, price_min, price_max):
    if kind == "prose_no_action":
        return "I'll open strong and see how the other side responds."
    p = min(own + 20, price_max) if seat == SELLER else max(own - 20, price_min)
    if kind == "missing_sentinel":
        return f"OFFER({p})"
    return f"#### OFFER({p})\n#### WALK"


def make_retry_dataset(n, seed=0, seat=SELLER, price_min=0, price_max=150):


    rows = []
    for i in range(n):
        draw = sample_draw(random.Random((seed << 34) | (i << 2)))
        kind = _INVALID_KINDS[i % len(_INVALID_KINDS)]
        own = draw.c if seat == SELLER else draw.v
        info = {
            "c": draw.c,
            "v": draw.v,
            "seat": seat,
            "source_row": i,
            "invalid_kind": kind,
            "price_min": price_min,
            "price_max": price_max,
        }
        prompt = [
            {"role": "system", "content": render_prompt(draw, seat, price_min, price_max)},
            {"role": "user", "content": "Round 1. You open."},
            {"role": "assistant", "content": _invalid_attempt(kind, seat, own, price_min, price_max)},
            {"role": "user", "content": RETRY_MESSAGE},
        ]
        rows.append({"prompt": prompt, "answer": "", "info": info})
    return Dataset.from_list(rows)


def opening_survival_reward(state, **kwargs):
    return 1.0 if state.get("probe_survived") else 0.0


class OpeningProbeEnv(vf.MultiTurnEnv):


    def __init__(self, dataset, retry_style="legacy", **kwargs):
        if retry_style not in ("legacy", "opening_aware"):
            raise ValueError(f"unknown retry_style: {retry_style!r}")
        rubric = vf.Rubric(funcs=[opening_survival_reward], weights=[1.0])
        super().__init__(dataset=dataset, rubric=rubric, max_turns=4, **kwargs)
        self.retry_style = retry_style

    async def env_response(self, messages, state, **kwargs):
        info = state["info"]
        valid = (
            parse_action(
                _last_assistant_text(messages), None, info["price_min"], info["price_max"]
            )
            is not None
        )
        if valid or state.get("probe_retried"):
            state["probe_survived"] = valid
            response = [vf.UserMessage(content="Done.")]
            state["final_env_response"] = response
            return response
        state["probe_retried"] = True


        msg = RETRY_MESSAGE_OPENING if self.retry_style == "opening_aware" else RETRY_MESSAGE
        return [vf.UserMessage(content=msg)]


def load_probe(
    task,
    n=64,
    seed=0,
    seat=SELLER,
    price_min=0,
    price_max=150,
    formula="negated",
    style="end",
    scoring="negated",
    termination="standard",
    retry_style="legacy",
    c_lo=C_LO,
    c_hi=C_HI,
    v_lo=V_LO,
    v_hi=V_HI,
    t_lo=None,
    t_hi=None,
):
    if task != "fullinfo" and (formula, style) != ("negated", "end"):
        raise ValueError(f"probe task {task!r} takes no formula/style")
    if task not in ("opening", "format_ctx") and (scoring, termination) != ("negated", "standard"):
        raise ValueError(f"probe task {task!r} takes no scoring/termination")
    if task != "opening" and retry_style != "legacy":
        raise ValueError(f"probe task {task!r} takes no retry_style")


    if task not in ("opening", "format_ctx") and (
        (c_lo, c_hi, v_lo, v_hi, t_lo, t_hi) != (C_LO, C_HI, V_LO, V_HI, None, None)
    ):
        raise ValueError(f"probe task {task!r} takes no draw-bound args")
    dk = dict(c_lo=c_lo, c_hi=c_hi, v_lo=v_lo, v_hi=v_hi, t_lo=t_lo, t_hi=t_hi)
    if task == "opening":
        return OpeningProbeEnv(
            dataset=make_format_ctx_dataset(
                n, seed, seat, price_min, price_max,
                scoring=scoring, termination=termination, **dk,
            ),
            retry_style=retry_style,
        )
    if task == "format":
        dataset = make_format_dataset(n, price_min, price_max)
        funcs = [format_reward]
    elif task == "format_ctx":
        dataset = make_format_ctx_dataset(
            n, seed, seat, price_min, price_max,
            scoring=scoring, termination=termination, **dk,
        )
        funcs = [format_reward]
    elif task == "retry":
        dataset = make_retry_dataset(n, seed, seat, price_min, price_max)
        funcs = [format_reward]
    elif task == "fullinfo":
        dataset = make_fullinfo_dataset(n, seed, seat, price_min, price_max, formula, style)
        funcs = [
            fullinfo_reward,
            valid_action_rate,
            offer_rate,
            reservation_safe_rate,
            correct_side_rate,
            opp_safe_rate,
            within_one_rate,
        ]
    else:
        raise ValueError(f"unknown probe task: {task!r}")
    rubric = vf.Rubric(funcs=funcs, weights=[1.0] + [0.0] * (len(funcs) - 1))
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
