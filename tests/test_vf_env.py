import asyncio

import pytest
import verifiers as vf

from market.env import BUYER, SELLER, Draw, Env, play
from market import bots
from market.vf_env import (
    _last_assistant_text,
    BargainingEnv,
    bargaining_reward,
    E_RANGES,
    load_environment,
    make_dataset,
    render_prompt,
)


def _info(c, v, horizon, seat=SELLER, opponent="conceder", seed=0):
    mode, param = horizon
    return {
        "c": c,
        "v": v,
        "horizon": {"mode": mode, "param": param},
        "seat": seat,
        "opponent": opponent,
        "seed": seed,
        "price_min": 0,
        "price_max": 150,
    }


def _env():

    return BargainingEnv(dataset=make_dataset(1, seed=0))


def _state(info):
    prompt = [vf.SystemMessage(content="sys")]
    return vf.State.for_task({"prompt": prompt, "info": info, "answer": ""})


async def _drive(env, state, learner_lines):
    await env.setup_state(state)
    messages = list(state["prompt"])
    for text in learner_lines:
        if state.get("final_env_response") is not None:
            break
        messages.append(vf.AssistantMessage(content=text))
        resp = await env.env_response(messages, state)
        messages.extend(resp)
    return messages


def _run(coro):
    return asyncio.run(coro)


def test_happy_path_deal_matches_direct_replay():
    env = _env()
    info = _info(50, 100, ("T", 6))
    state = _state(info)
    _run(_drive(env, state, ["#### OFFER(90)", "#### ACCEPT"]))

    assert state.get("final_env_response") is not None
    market_env = state["market"]["env"]
    assert market_env.outcome == ("deal", 58)

    reward = bargaining_reward(state=state)

    ref = Env(Draw(50, 100, ("T", 6)), __import__("random").Random(0), first_mover=SELLER)
    r_ref = play(ref, _script(["#### OFFER(90)", "#### ACCEPT"]), bots.conceder(BUYER, 100, 0))
    assert reward == r_ref[0]
    assert reward < 0


def test_invalid_once_returns_retry_and_opponent_does_not_move():
    env = _env()
    state = _state(_info(50, 100, ("T", 6)))
    asyncio.run(env.setup_state(state))
    messages = list(state["prompt"]) + [vf.AssistantMessage(content="no action here")]
    resp = asyncio.run(env.env_response(messages, state))

    market_env = state["market"]["env"]
    assert state.get("final_env_response") is None
    assert market_env.turn == SELLER
    assert market_env.last_offer is None
    assert "retry" in resp[0].content.lower()


def test_invalid_twice_forfeits_learner_at_minus_one():
    env = _env()
    state = _state(_info(50, 100, ("T", 6)))
    _run(_drive(env, state, ["garbage", "still garbage"]))

    assert state.get("final_env_response") is not None
    assert state["market"]["env"].outcome == ("forfeit", SELLER)
    assert bargaining_reward(state=state) == -1.0


def test_opponent_accepts_learner_offer_ends_episode():
    env = _env()
    state = _state(_info(50, 100, ("T", 6)))
    _run(_drive(env, state, ["#### OFFER(55)"]))

    market_env = state["market"]["env"]
    assert state.get("final_env_response") is not None
    assert market_env.outcome == ("deal", 55)
    assert "ACCEPT" in state["final_env_response"][0].content
    reward = bargaining_reward(state=state)
    ref = Env(Draw(50, 100, ("T", 6)), __import__("random").Random(0), first_mover=SELLER)
    r_ref = play(ref, _script(["#### OFFER(55)"]), bots.conceder(BUYER, 100, 0))
    assert reward == r_ref[0]


def test_reward_zero_for_walk_and_timeout():
    env = _env()
    state = _state(_info(50, 100, ("T", 6)))
    _run(_drive(env, state, ["#### WALK"]))
    assert state["market"]["env"].outcome == ("walk", SELLER)
    assert bargaining_reward(state=state) == 0.0

    state2 = _state(_info(50, 100, ("T", 3)))
    _run(_drive(env, state2, ["#### OFFER(120)", "#### OFFER(120)"]))
    assert state2["market"]["env"].outcome == ("timeout",)
    assert bargaining_reward(state=state2) == 0.0


def test_prompt_shows_own_value_not_opponents():
    draw = Draw(c=62, v=99, horizon=("T", 11))
    seller_prompt = render_prompt(draw, SELLER, 0, 150)
    assert "c = 62" in seller_prompt
    assert "Your value" not in seller_prompt
    assert "99" not in seller_prompt
    assert "PRIORS" in seller_prompt
    assert "HORIZON" in seller_prompt
    assert "T = 11" in seller_prompt
    assert "RULES" in seller_prompt
    assert "OFFER" in seller_prompt

    buyer_prompt = render_prompt(draw, BUYER, 0, 150)
    assert "v = 99" in buyer_prompt
    assert "62" not in buyer_prompt


def test_load_environment_builds_bargaining_env():
    env = load_environment(n=4, seed=1)
    assert isinstance(env, BargainingEnv)
    assert len(env.get_dataset()) == 4


def _script(lines):
    it = iter(lines)

    class _S:
        def act(self, env):
            return next(it)

    return _S()


def test_prompt_no_zopa_prob_matches_priors():
    draw = Draw(c=60, v=70, horizon=("T", 8))
    prompt = render_prompt(draw, SELLER, 0, 150)
    expected = sum(v <= c for c in range(40, 81) for v in range(50, 101)) / (41 * 51)
    assert f"P(no-ZOPA) = {expected:.2f}" in prompt
    assert "0.15" not in prompt


def test_last_assistant_text_none_and_structured_content():
    assert _last_assistant_text([{"role": "assistant", "content": None}]) == ""
    structured = [{"role": "assistant", "content": [{"type": "text", "text": "#### OFFER(70)"}]}]
    assert _last_assistant_text(structured) == "#### OFFER(70)"


def test_unstepped_final_action_is_flushed_before_scoring():


    env = _env()
    state = _state(_info(50, 100, ("T", 6)))
    messages = _run(_drive(env, state, ["#### OFFER(90)"]))
    market_env = state["market"]["env"]
    assert not market_env.done

    state["trajectory"].append(
        {"prompt": [], "completion": [vf.AssistantMessage(content="#### OFFER(90)")]}
    )
    state["trajectory"].append(
        {"prompt": messages, "completion": [vf.AssistantMessage(content="#### ACCEPT")]}
    )
    assert any(getattr(h, "__name__", "") == "flush_pending_action" for h in env._cleanup_handlers)
    _run(env.flush_pending_action(state))
    assert market_env.outcome[0] == "deal"
    assert bargaining_reward(state=state) != 0.0


def test_flush_of_nonterminal_pending_action_leaves_reward_zero():
    env = _env()
    state = _state(_info(50, 100, ("T", 6)))
    messages = _run(_drive(env, state, ["#### OFFER(90)"]))
    state["trajectory"].append(
        {"prompt": [], "completion": [vf.AssistantMessage(content="#### OFFER(90)")]}
    )
    state["trajectory"].append(
        {"prompt": messages, "completion": [vf.AssistantMessage(content="#### OFFER(85)")]}
    )
    _run(env.flush_pending_action(state))
    assert not state["market"]["env"].done
    assert bargaining_reward(state=state) == 0.0


def test_opponent_params_randomized_per_row_and_hidden():
    ds = make_dataset(20, seed=3)
    es = {row["info"]["opp_e"] for row in ds}
    anchors = {row["info"]["opp_anchor"] for row in ds}
    assert len(es) > 5 and len(anchors) > 5
    for row in ds:
        content = row["prompt"][0]["content"]
        assert str(row["info"]["opp_anchor"]) not in content.split("PRIORS")[0]
        assert "opp_e" not in content and "anchor" not in content


def test_random_threshold_params_sampled_and_hidden():


    ds = make_dataset(30, seed=5, opponent="random_threshold")
    thresholds = set()
    for row in ds:
        info = row["info"]
        assert info["opp_e"] is None
        assert 40 <= info["opp_anchor"] <= info["v"]
        thresholds.add(info["opp_anchor"])


        draw = Draw(c=info["c"], v=info["v"], horizon=("T", int(info["horizon"]["param"])))
        if info["horizon"]["mode"] == "geom":
            draw = Draw(c=info["c"], v=info["v"], horizon=("geom", info["horizon"]["param"]))
        content = row["prompt"][0]["content"]
        assert content == render_prompt(
            draw, info["seat"], info["price_min"], info["price_max"]
        )
        assert "threshold" not in content.lower()
    assert len(thresholds) > 5


def test_prompt_states_scoring_and_has_no_numeric_example():


    import re

    draw = Draw(c=60, v=90, horizon=("T", 8))
    for seat, side in [(SELLER, "above"), (BUYER, "below")]:
        p = render_prompt(draw, seat, 0, 150)
        assert (
            "SCORING (public): a deal at price p scores (2p - v - c)/(v - c), "
            "clamped to [-1, 1]" in p
        )
        assert f"positive only if p is {side} the midpoint" in p

        assert "If v <= c (no ZOPA), a deal scores +1/-1 by the same midpoint rule" in p

        assert "WALK, deadline, and breakdown score 0" in p
        assert "Forfeiting by invalid output scores you -1" in p
        assert "A deal can score worse than walking away." in p


        assert "Example" not in p
        assert not re.search(r"#### OFFER\(\d", p)


def test_prompt_scoring_matches_verifier_on_no_zopa_and_degenerate_draws():


    from market.env import _deal_rewards


    assert _deal_rewards(80, 50, 70) == (1.0, -1.0)
    assert _deal_rewards(80, 50, 60) == (-1.0, 1.0)
    assert _deal_rewards(80, 50, 65) == (0.0, 0.0)

    assert _deal_rewards(60, 60, 60) == (0.0, 0.0)
    assert _deal_rewards(60, 60, 61) == (1.0, -1.0)

    assert _deal_rewards(50, 90, 80) == (0.5, -0.5)


def test_mix_opponent_samples_per_episode():


    ds = make_dataset(60, seed=7, opponent="mix")
    names = {row["info"]["opponent"] for row in ds}
    assert names == {"boulware", "conceder", "random_threshold"}
    for row in ds:
        info = row["info"]
        if info["opponent"] == "random_threshold":
            assert info["opp_e"] is None
        else:
            lo, hi = E_RANGES[info["opponent"]]
            assert lo <= info["opp_e"] <= hi


def test_mix_weights_skew_opponent_sampling():
    ds = make_dataset(40, seed=7, opponent="mix", mix_weights={"conceder": 1.0})
    assert {row["info"]["opponent"] for row in ds} == {"conceder"}
    ds = make_dataset(
        60, seed=7, opponent="mix",
        mix_weights={"conceder": 0.5, "boulware": 0.25, "random_threshold": 0.25},
    )
    counts = {}
    for row in ds:
        counts[row["info"]["opponent"]] = counts.get(row["info"]["opponent"], 0) + 1
    assert counts["conceder"] > counts["boulware"]
    assert counts["conceder"] > counts["random_threshold"]


def test_unweighted_mix_exactly_balanced():
    ds = make_dataset(9, seed=3, opponent="mix")
    counts = {}
    for row in ds:
        counts[row["info"]["opponent"]] = counts.get(row["info"]["opponent"], 0) + 1
    assert counts == {"boulware": 3, "conceder": 3, "random_threshold": 3}


def test_unweighted_mix_assignment_shuffled_and_seat_shared():


    names = sorted(["boulware", "conceder", "random_threshold"])
    seq = [r["info"]["opponent"] for r in make_dataset(30, seed=999, opponent="mix")]
    assert seq != [names[i % 3] for i in range(30)]
    again = [r["info"]["opponent"] for r in make_dataset(30, seed=999, opponent="mix")]
    assert seq == again
    buyer = [
        r["info"]["opponent"]
        for r in make_dataset(30, seed=999, opponent="mix", seat=BUYER)
    ]
    assert seq == buyer


def test_outcome_metric_funcs():
    import random as _random

    from market.vf_env import (
        deal_rate,
        negative_deal_rate,
        positive_deal_rate,
        walk_rate,
        zopa_rate,
        zopa_walk_rate,
    )

    draw = Draw(c=50, v=90, horizon=("T", 10))
    walk_env = Env(draw, _random.Random(0), first_mover=SELLER)
    walk_env.step("#### WALK")
    state = {"market": {"env": walk_env, "seat": SELLER}, "info": {"c": 50, "v": 90}}
    assert walk_rate(state) == 1.0
    assert deal_rate(state) == 0.0
    assert zopa_rate(state) == 1.0
    assert zopa_walk_rate(state) == 1.0

    deal_env = Env(draw, _random.Random(0), first_mover=SELLER)
    deal_env.step("#### OFFER(80)")
    deal_env.step("#### ACCEPT")
    state = {"market": {"env": deal_env, "seat": SELLER}, "info": {"c": 50, "v": 90}}
    assert deal_rate(state) == 1.0
    assert positive_deal_rate(state) == 1.0
    assert negative_deal_rate(state) == 0.0
    assert walk_rate(state) == 0.0
    assert zopa_walk_rate(state) == 0.0


    state = {"market": {"env": deal_env, "seat": BUYER}, "info": {"c": 50, "v": 90}}
    assert positive_deal_rate(state) == 0.0
    assert negative_deal_rate(state) == 1.0


def test_outcome_metric_funcs_partition_terminals():
    import random as _random

    from market.vf_env import (
        breakdown_rate,
        deal_rate,
        forfeit_rate,
        timeout_rate,
        walk_rate,
    )

    partition = [deal_rate, walk_rate, forfeit_rate, timeout_rate, breakdown_rate]

    def flags(env):
        state = {"market": {"env": env, "seat": SELLER}, "info": {"c": 50, "v": 90}}
        return [f(state) for f in partition]

    draw = Draw(c=50, v=90, horizon=("T", 2))

    forfeit_env = Env(draw, _random.Random(0), first_mover=SELLER)
    forfeit_env.step("garbage")
    forfeit_env.step("still garbage")
    assert forfeit_env.done and forfeit_env.outcome[0] == "forfeit"
    assert flags(forfeit_env) == [0.0, 0.0, 1.0, 0.0, 0.0]

    timeout_env = Env(draw, _random.Random(0), first_mover=SELLER)
    while not timeout_env.done:
        timeout_env.step("#### OFFER(100)")
    assert timeout_env.outcome[0] == "timeout"
    assert flags(timeout_env) == [0.0, 0.0, 0.0, 1.0, 0.0]

    geom = Draw(c=50, v=90, horizon=("geom", 0.0))
    breakdown_env = Env(geom, _random.Random(0), first_mover=SELLER)
    while not breakdown_env.done:
        breakdown_env.step("#### OFFER(100)")
    assert breakdown_env.outcome[0] == "breakdown"
    assert flags(breakdown_env) == [0.0, 0.0, 0.0, 0.0, 1.0]


    open_env = Env(draw, _random.Random(0), first_mover=SELLER)
    open_env.step("#### OFFER(100)")
    assert not open_env.done
    assert flags(open_env) == [0.0, 0.0, 0.0, 0.0, 0.0]


def _game_keys(ds):
    return [
        (r["info"]["c"], r["info"]["v"], r["info"]["horizon"]["mode"],
         r["info"]["horizon"]["param"], r["info"]["seed"])
        for r in ds
    ]


def test_mix_weights_do_not_shift_game_draws():


    uniform = make_dataset(30, seed=11, opponent="mix")
    weighted = make_dataset(30, seed=11, opponent="mix", mix_weights={"conceder": 1.0})
    fixed = make_dataset(30, seed=11, opponent="boulware")
    assert _game_keys(uniform) == _game_keys(weighted) == _game_keys(fixed)


def test_same_seed_pairs_games_across_seats():

    seller = make_dataset(20, seed=999, opponent="mix", seat=SELLER)
    buyer = make_dataset(20, seed=999, opponent="mix", seat=BUYER)
    assert _game_keys(seller) == _game_keys(buyer)
    assert [r["info"]["opponent"] for r in seller] == [r["info"]["opponent"] for r in buyer]


def test_mix_weights_validation():
    import pytest

    with pytest.raises(ValueError, match="requires opponent='mix'"):
        make_dataset(2, opponent="boulware", mix_weights={"conceder": 1.0})
    with pytest.raises(ValueError, match="unknown opponents"):
        make_dataset(2, opponent="mix", mix_weights={"tit_for_tat": 1.0})
    with pytest.raises(ValueError, match="finite and non-negative"):
        make_dataset(2, opponent="mix", mix_weights={"conceder": -1.0})
    with pytest.raises(ValueError, match="positive sum"):
        make_dataset(2, opponent="mix", mix_weights={"conceder": 0.0})


def test_random_threshold_opponent_constructed_from_info():
    info = _info(50, 100, ("T", 6), opponent="random_threshold")
    info["opp_e"] = None
    info["opp_anchor"] = 77
    env = _env()
    state = _state(info)
    _run(env.setup_state(state))
    opponent = state["market"]["opponent"]
    assert isinstance(opponent, bots.RandomThreshold)
    assert opponent.threshold == 77


def test_opponent_uses_sampled_params():
    info = _info(50, 100, ("T", 6))
    info["opp_e"] = 3.0
    info["opp_anchor"] = 25
    env = _env()
    state = _state(info)
    _run(env.setup_state(state))
    opponent = state["market"]["opponent"]
    assert opponent.e == 3.0 and opponent.anchor == 25


_PRE_0112_PROMPT_SHA = {
    (SELLER, ("T", 6)): "5764732dd8ad0ca1a05335529d959bd10cd64faf45208ece644369e97c3c143e",
    (SELLER, ("geom", 0.92)): "f8b889b0c0aca960c0d8a64da9e2a760a9e776f2e0eab147ed31913d9820c108",
    (BUYER, ("T", 6)): "50ef02b96fc51d40550bb329f979d5e279d8d4eb6442d0444d475fa306b9f60d",
    (BUYER, ("geom", 0.92)): "11792e336fd8f3a1d538a02afa6dbe062cfdaecdd76dd5d4a4f27f73c8e76ca8",
}


def test_prompt_variant_defaults_byte_identical():
    import hashlib

    for (seat, horizon), expected in _PRE_0112_PROMPT_SHA.items():
        draw = Draw(50 if horizon[0] == "T" else 63, 100 if horizon[0] == "T" else 77, horizon)
        default = render_prompt(draw, seat, 0, 150)
        assert hashlib.sha256(default.encode()).hexdigest() == expected
        assert default == render_prompt(
            draw, seat, 0, 150, scoring="negated", termination="standard"
        )


def test_prompt_rolelocal_scoring_states_own_formula():
    draw = Draw(50, 100, ("T", 6))
    buyer = render_prompt(draw, BUYER, 0, 150, scoring="rolelocal")
    assert "(v + c - 2p)/(v - c)" in buyer
    assert "negative for the buyer" not in buyer
    assert "zero-sum" in buyer
    seller = render_prompt(draw, SELLER, 0, 150, scoring="rolelocal")
    assert "(2p - v - c)/(v - c)" in seller
    assert "negative for the buyer" not in seller

    base = render_prompt(draw, BUYER, 0, 150)
    assert buyer.split("SCORING")[0] == base.split("SCORING")[0]
    assert buyer.split("Your score is positive")[1] == base.split("Your score is positive")[1]


def test_prompt_reserve_termination_appends_guidance_only():
    draw = Draw(50, 100, ("T", 6))
    base = render_prompt(draw, SELLER, 0, 150)
    reserve = render_prompt(draw, SELLER, 0, 150, termination="reserve")
    assert reserve.startswith(base)
    tail = reserve[len(base):]
    assert "leave room to finish" in tail
    assert "Think briefly" in tail


def test_prompt_unknown_variant_raises():
    import pytest

    draw = Draw(50, 100, ("T", 6))
    with pytest.raises(ValueError, match="unknown scoring"):
        render_prompt(draw, SELLER, 0, 150, scoring="direct")
    with pytest.raises(ValueError, match="unknown termination"):
        render_prompt(draw, SELLER, 0, 150, termination="short")


def test_retry_style_opening_aware_message_only_on_opening():
    from market.vf_env import RETRY_MESSAGE, RETRY_MESSAGE_OPENING

    env = BargainingEnv(dataset=make_dataset(1, seed=0), retry_style="opening_aware")
    state = _state(_info(50, 100, ("T", 6)))
    asyncio.run(env.setup_state(state))

    messages = list(state["prompt"]) + [vf.AssistantMessage(content="no action")]
    resp = asyncio.run(env.env_response(messages, state))
    assert resp[0].content == RETRY_MESSAGE_OPENING
    assert "ACCEPT is not available" in resp[0].content


    messages = messages[:-1] + [vf.AssistantMessage(content="#### OFFER(140)")]
    asyncio.run(env.env_response(messages, state))
    assert state["market"]["env"].last_offer is not None
    messages = messages + [vf.AssistantMessage(content="rambling")]
    resp = asyncio.run(env.env_response(messages, state))
    assert resp[0].content == RETRY_MESSAGE


def test_retry_style_legacy_default_and_unknown_raises():
    import pytest

    from market.vf_env import RETRY_MESSAGE

    env = BargainingEnv(dataset=make_dataset(1, seed=0))
    state = _state(_info(50, 100, ("T", 6)))
    asyncio.run(env.setup_state(state))
    messages = list(state["prompt"]) + [vf.AssistantMessage(content="no action")]
    resp = asyncio.run(env.env_response(messages, state))
    assert resp[0].content == RETRY_MESSAGE
    with pytest.raises(ValueError, match="unknown retry_style"):
        BargainingEnv(dataset=make_dataset(1, seed=0), retry_style="opening")


def test_make_dataset_variant_prompts_flow_through():
    base = make_dataset(3, seed=0, opponent="conceder", seat=BUYER)
    var = make_dataset(
        3, seed=0, opponent="conceder", seat=BUYER, scoring="rolelocal", termination="reserve"
    )
    for a, b in zip(base, var):
        assert a["info"] == b["info"]
        assert "(v + c - 2p)/(v - c)" in b["prompt"][0]["content"]
        assert "leave room to finish" in b["prompt"][0]["content"]
        assert a["prompt"][0]["content"] != b["prompt"][0]["content"]


def test_draw_bounds_shape_draws_and_priors_line():
    ds = make_dataset(40, seed=2, opponent="mix", c_lo=40, c_hi=60, v_lo=90, v_hi=110)
    for r in ds:
        info = r["info"]
        assert 40 <= info["c"] <= 60 and 90 <= info["v"] <= 110
        assert info["v"] - info["c"] >= 30
        assert "c ~ U[40,60], v ~ U[90,110]" in r["prompt"][0]["content"]
        assert "P(no-ZOPA) = 0.00" in r["prompt"][0]["content"]


def test_t_range_forces_short_revealed_horizon():
    ds = make_dataset(30, seed=2, opponent="mix", t_lo=2, t_hi=6)
    for r in ds:
        h = r["info"]["horizon"]
        assert h["mode"] == "T" and 2 <= h["param"] <= 6
        assert "revealed, T = " in r["prompt"][0]["content"]


def test_default_bounds_unchanged():
    assert make_dataset(5, seed=0)[0] == make_dataset(
        5, seed=0, c_lo=40, c_hi=80, v_lo=50, v_hi=100
    )[0]


def test_draw_knob_validation():
    with pytest.raises(ValueError, match="together"):
        make_dataset(2, seed=0, t_lo=2)
    with pytest.raises(ValueError, match="bad t range"):
        make_dataset(2, seed=0, t_lo=1, t_hi=6)
    with pytest.raises(ValueError, match="bad draw bounds"):
        make_dataset(2, seed=0, c_lo=80, c_hi=40)


    load_environment(n=2, task="opening", c_lo=40, c_hi=60)
    with pytest.raises(ValueError, match="no draw-bound args"):
        load_environment(n=2, task="fullinfo", c_lo=40, c_hi=60)


def test_load_environment_threads_draw_knobs():
    env = load_environment(n=4, seed=3, opponent="mix", c_lo=40, c_hi=60,
                           v_lo=90, v_hi=110, t_lo=2, t_hi=4)
    for r in env.dataset:
        assert r["info"]["v"] - r["info"]["c"] >= 30
        assert r["info"]["horizon"]["mode"] == "T" and r["info"]["horizon"]["param"] <= 4


def test_default_datasets_byte_identical_to_0112_golden():


    import hashlib
    import json
    import pathlib

    golden = json.loads(
        (pathlib.Path(__file__).parent / "fixtures" / "golden_0112_dataset_hashes.json")
        .read_text()
    )
    cells = {
        "mix-seller-64-s0": dict(n=64, seed=0, opponent="mix", seat="seller"),
        "mix-buyer-64-s1": dict(n=64, seed=1, opponent="mix", seat="buyer"),
        "conceder-buyer-32-s7": dict(n=32, seed=7, opponent="conceder", seat="buyer"),
        "mix-seller-eval-63-s999": dict(n=63, seed=999, opponent="mix", seat="seller"),
    }
    for name, kw in cells.items():
        blob = json.dumps(make_dataset(**kw).to_list(), sort_keys=True, default=str).encode()
        assert hashlib.sha256(blob).hexdigest() == golden[name], name


def test_draw_bounds_cross_support_validation_16b15():

    with pytest.raises(ValueError, match="c_lo <= v_lo"):
        make_dataset(2, seed=0, opponent="mix", c_lo=90, c_hi=110, v_lo=40, v_hi=60)
    with pytest.raises(ValueError, match="c_hi <= v_hi"):
        make_dataset(2, seed=0, opponent="mix", c_lo=40, c_hi=95, v_lo=50, v_hi=90)
    with pytest.raises(ValueError, match="price grid"):
        make_dataset(2, seed=0, opponent="mix", v_lo=140, v_hi=160)

    make_dataset(2, seed=0, opponent="mix", c_lo=40, c_hi=60, v_lo=90, v_hi=110)
    make_dataset(2, seed=0, opponent="mix")
