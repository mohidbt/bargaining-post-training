
import asyncio
import re

import pytest
import verifiers as vf

from market.env import BUYER, SELLER, _deal_rewards, parse_action
from market.vf_env import RETRY_MESSAGE, load_environment, make_dataset
from market.vf_probe import (
    OpeningProbeEnv,
    correct_side_rate,
    load_probe,
    format_reward,
    fullinfo_reward,
    make_format_ctx_dataset,
    make_format_dataset,
    make_fullinfo_dataset,
    make_retry_dataset,
    offer_rate,
    opening_survival_reward,
    opp_safe_rate,
    within_one_rate,
    render_format_prompt,
    render_fullinfo_prompt,
    reservation_safe_rate,
    valid_action_rate,
)


def completion(text):
    return [{"role": "assistant", "content": text}]


INFO = {"seat": SELLER, "c": 60, "v": 90, "price_min": 0, "price_max": 150}


class TestFormatProbe:
    def test_valid_offer_and_walk_pass(self):
        assert format_reward(completion("#### OFFER(75)"), INFO) == 1.0
        assert format_reward(completion("thoughts first\n#### WALK"), INFO) == 1.0

    def test_accept_invalid_without_standing_offer(self):
        assert format_reward(completion("#### ACCEPT"), INFO) == 0.0

    def test_garbage_two_actions_and_off_grid_fail(self):
        assert format_reward(completion("I offer 75"), INFO) == 0.0
        assert format_reward(completion("#### OFFER(75)\n#### WALK"), INFO) == 0.0
        assert format_reward(completion("#### OFFER(151)"), INFO) == 0.0

    def test_think_block_stripped(self):
        text = "<think>#### OFFER(75)</think>\n#### WALK"
        assert format_reward(completion(text), INFO) == 1.0

    def test_trailing_prose_tolerated_like_real_env(self):


        assert format_reward(completion("#### WALK\nand a remark after"), INFO) == 1.0

    def test_prompt_has_no_example_price_and_no_accept(self):
        prompt = render_format_prompt(0, 150)
        assert not re.search(r"####\s*OFFER\(\d", prompt)
        assert "ACCEPT is not available" in prompt

    def test_dataset_rows_identical(self):
        ds = make_format_dataset(5)
        assert len(ds) == 5
        assert ds[0] == ds[4]


class TestFullinfoRewards:
    def test_seller_exact_target_passes(self):

        c = completion("#### OFFER(76)")
        assert fullinfo_reward(c, INFO) == 1.0
        assert valid_action_rate(c, INFO) == 1.0
        assert offer_rate(c, INFO) == 1.0
        assert reservation_safe_rate(c, INFO) == 1.0
        assert correct_side_rate(c, INFO) == 1.0
        assert opp_safe_rate(c, INFO) == 1.0
        assert within_one_rate(c, INFO) == 1.0

    def test_seller_positive_but_not_lowest_fails(self):
        c = completion("#### OFFER(80)")
        assert correct_side_rate(c, INFO) == 1.0
        assert opp_safe_rate(c, INFO) == 1.0
        assert fullinfo_reward(c, INFO) == 0.0

        c = completion("#### OFFER(75)")
        assert within_one_rate(c, INFO) == 1.0
        assert fullinfo_reward(c, INFO) == 0.0

    def test_value_blind_and_copy_policies_fail(self):


        assert fullinfo_reward(completion("#### OFFER(150)"), INFO) == 0.0
        assert fullinfo_reward(completion("#### OFFER(90)"), INFO) == 0.0
        info = dict(INFO, seat=BUYER)
        assert fullinfo_reward(completion("#### OFFER(0)"), info) == 0.0
        assert fullinfo_reward(completion("#### OFFER(60)"), info) == 0.0

    def test_value_blind_ceilings_on_actual_datasets(self):


        expected = {SELLER: (27, 16, 251), BUYER: (26, 16, 251)}
        for seat in (SELLER, BUYER):
            infos = [r["info"] for r in make_fullinfo_dataset(512, seed=0, seat=seat)]
            best = max(
                sum(fullinfo_reward(completion(f"#### OFFER({p})"), i) for i in infos)
                for p in range(0, 151)
            )
            disclosed = "v" if seat == SELLER else "c"
            copy = sum(
                fullinfo_reward(completion(f"#### OFFER({i[disclosed]})"), i) for i in infos
            )
            rounded = sum(
                fullinfo_reward(
                    completion(
                        f"#### OFFER({(i['c'] + i['v'] + 1) // 2 if seat == SELLER else (i['c'] + i['v']) // 2})"
                    ),
                    i,
                )
                for i in infos
            )
            assert (best, copy, rounded) == expected[seat]

    def test_seller_profitable_but_below_midpoint_fails(self):
        c = completion("#### OFFER(70)")
        assert fullinfo_reward(c, INFO) == 0.0
        assert reservation_safe_rate(c, INFO) == 1.0
        assert correct_side_rate(c, INFO) == 0.0

    def test_seller_reservation_violation(self):
        c = completion("#### OFFER(50)")
        assert reservation_safe_rate(c, INFO) == 0.0
        assert fullinfo_reward(c, INFO) == 0.0

    def test_buyer_side_flipped(self):
        info = dict(INFO, seat=BUYER)

        assert fullinfo_reward(completion("#### OFFER(74)"), info) == 1.0
        assert fullinfo_reward(completion("#### OFFER(70)"), info) == 0.0
        assert fullinfo_reward(completion("#### OFFER(80)"), info) == 0.0
        assert within_one_rate(completion("#### OFFER(75)"), info) == 1.0

        c = completion("#### OFFER(95)")
        assert reservation_safe_rate(c, info) == 0.0
        assert fullinfo_reward(c, info) == 0.0

    def test_odd_sum_targets(self):


        info = dict(INFO, c=59, v=60)
        assert fullinfo_reward(completion("#### OFFER(60)"), info) == 1.0
        assert fullinfo_reward(completion("#### OFFER(59)"), dict(info, seat=BUYER)) == 1.0

    def test_walk_is_valid_but_scores_zero(self):
        c = completion("#### WALK")
        assert valid_action_rate(c, INFO) == 1.0
        assert offer_rate(c, INFO) == 0.0
        assert fullinfo_reward(c, INFO) == 0.0

    def test_garbage_scores_zero_everywhere(self):
        c = completion("the price should be 80")
        for f in (
            fullinfo_reward,
            valid_action_rate,
            offer_rate,
            reservation_safe_rate,
            correct_side_rate,
            opp_safe_rate,
            within_one_rate,
        ):
            assert f(c, INFO) == 0.0

    @pytest.mark.parametrize("seat", [SELLER, BUYER])
    def test_target_is_boundary_optimum_under_real_verifier(self, seat):


        idx = 0 if seat == SELLER else 1
        step = -1 if seat == SELLER else 1
        for c, v in [(60, 90), (59, 60), (40, 100), (79, 80), (44, 99)]:
            info = dict(INFO, c=c, v=v, seat=seat)
            targets = [
                p
                for p in range(0, 151)
                if fullinfo_reward(completion(f"#### OFFER({p})"), info) == 1.0
            ]
            assert len(targets) == 1
            t = targets[0]
            assert _deal_rewards(c, v, t)[idx] > 0
            assert _deal_rewards(c, v, t + step)[idx] <= 0


class TestFullinfoDataset:
    def test_zopa_only_and_paired_with_bargaining_rows(self):
        bargain = make_dataset(60, seed=0, opponent="boulware", seat=SELLER)
        probe = make_fullinfo_dataset(30, seed=0)
        zopa_rows = [r["info"] for r in bargain if r["info"]["v"] > r["info"]["c"]]
        assert len(probe) == 30
        for got, want in zip(probe, zopa_rows):
            assert got["info"]["v"] > got["info"]["c"]
            assert (got["info"]["c"], got["info"]["v"]) == (want["c"], want["v"])
        source_rows = [r["info"]["source_row"] for r in probe]
        assert source_rows == sorted(source_rows)

    def test_deterministic_and_seat_changes_prompt_not_draws(self):
        a = make_fullinfo_dataset(10, seed=3, seat=SELLER)
        b = make_fullinfo_dataset(10, seed=3, seat=BUYER)
        assert [r["info"]["c"] for r in a] == [r["info"]["c"] for r in b]
        assert a[0]["prompt"] != b[0]["prompt"]
        assert make_fullinfo_dataset(10, seed=3, seat=SELLER)[9] == a[9]

    def test_prompt_discloses_both_values_no_example(self):
        for seat, side, extreme in ((SELLER, "above", "LOWEST"), (BUYER, "below", "HIGHEST")):
            prompt = render_fullinfo_prompt(55, 85, seat, 0, 150)
            assert "c = 55" in prompt and "v = 85" in prompt
            assert f"positive only if p is {side} the midpoint" in prompt
            assert "(2p - v - c)/(v - c)" in prompt
            assert f"compute the {extreme} integer price" in prompt
            assert "submit exactly that price" in prompt
            assert not re.search(r"####\s*OFFER\(\d", prompt)


class TestFullinfoVariants:
    def test_default_is_round7c_prompt_verbatim(self):

        assert render_fullinfo_prompt(55, 85, BUYER, 0, 150) == render_fullinfo_prompt(
            55, 85, BUYER, 0, 150, formula="negated", style="end"
        )
        assert "the negative for the buyer" in render_fullinfo_prompt(55, 85, BUYER, 0, 150)

    def test_rolelocal_buyer_gets_direct_formula_no_negation(self):
        prompt = render_fullinfo_prompt(55, 85, BUYER, 0, 150, formula="rolelocal")
        assert "(v + c - 2p)/(v - c)" in prompt
        assert "negative for the buyer" not in prompt
        assert "zero-sum" in prompt
        seller = render_fullinfo_prompt(55, 85, SELLER, 0, 150, formula="rolelocal")
        assert "(2p - v - c)/(v - c)" in seller

    def test_style_first_demands_immediate_line(self):
        prompt = render_fullinfo_prompt(55, 85, SELLER, 0, 150, style="first")
        assert "MUST be ONLY the single line" in prompt
        assert "MUST end" not in prompt

    def test_variants_share_draws_targets_and_reward(self):
        base = make_fullinfo_dataset(20, seed=0, seat=BUYER)
        var = make_fullinfo_dataset(20, seed=0, seat=BUYER, formula="rolelocal", style="first")
        for a, b in zip(base, var):
            assert a["info"] == b["info"]
            assert a["prompt"] != b["prompt"]

        info = var[0]["info"]
        s = info["c"] + info["v"]
        target = (s + 1) // 2 - 1
        assert fullinfo_reward(completion(f"#### OFFER({target})"), info) == 1.0

    def test_unknown_formula_and_style_raise(self):
        with pytest.raises(ValueError, match="unknown formula"):
            render_fullinfo_prompt(55, 85, SELLER, 0, 150, formula="direct")
        with pytest.raises(ValueError, match="unknown style"):
            render_fullinfo_prompt(55, 85, SELLER, 0, 150, style="only")

    def test_negexplicit_buyer_direct_negated_formula(self):
        prompt = render_fullinfo_prompt(55, 85, BUYER, 0, 150, formula="negexplicit")
        assert "-((2p - v - c)/(v - c))" in prompt
        assert "for you" in prompt and "zero-sum" in prompt
        assert "negative for the buyer" not in prompt

        rolelocal = render_fullinfo_prompt(55, 85, BUYER, 0, 150, formula="rolelocal")
        assert prompt.replace("-((2p - v - c)/(v - c))", "(v + c - 2p)/(v - c)") == rolelocal

    def test_negexplicit_is_buyer_only(self):
        with pytest.raises(ValueError, match="buyer-only"):
            render_fullinfo_prompt(55, 85, SELLER, 0, 150, formula="negexplicit")

    def test_noformula_has_no_formula_but_keeps_midpoint_task(self):
        prompt = render_fullinfo_prompt(55, 85, BUYER, 0, 150, formula="noformula")
        assert "(v - c)" not in prompt and "2p" not in prompt
        assert "midpoint of c and v" in prompt
        assert "HIGHEST integer price" in prompt

    def test_new_variants_share_draws_and_reward(self):
        base = make_fullinfo_dataset(10, seed=0, seat=BUYER)
        for f in ("negexplicit", "noformula"):
            var = make_fullinfo_dataset(10, seed=0, seat=BUYER, formula=f)
            for a, b in zip(base, var):
                assert a["info"] == b["info"]
                assert a["prompt"] != b["prompt"]
        info = base[0]["info"]
        target = (info["c"] + info["v"] + 1) // 2 - 1
        assert fullinfo_reward(completion(f"#### OFFER({target})"), info) == 1.0

    def test_noformula_target_unique_and_in_grid_all_draws(self):


        rows = make_fullinfo_dataset(512, seed=0, seat=BUYER, formula="noformula")
        for r in rows:
            info = r["info"]
            c, v = info["c"], info["v"]
            target = (c + v + 1) // 2 - 1
            assert info["price_min"] <= target <= info["price_max"]
            assert 2 * target < c + v
            assert 2 * (target + 1) >= c + v
            assert fullinfo_reward(completion(f"#### OFFER({target})"), info) == 1.0
            assert fullinfo_reward(completion(f"#### OFFER({target + 1})"), info) == 0.0

    def test_seed7_dataset_sane_both_seats(self):


        s0 = make_fullinfo_dataset(64, seed=0, seat=SELLER)
        for seat in (SELLER, BUYER):
            rows = make_fullinfo_dataset(64, seed=7, seat=seat)
            assert all(r["info"]["v"] > r["info"]["c"] for r in rows)
            assert any(
                (a["info"]["c"], a["info"]["v"]) != (b["info"]["c"], b["info"]["v"])
                for a, b in zip(s0, rows)
            )


class TestFormatCtxProbe:
    def test_prompt_is_real_bargaining_opening(self):
        bargain = make_dataset(10, seed=0, opponent="boulware", seat=SELLER)
        probe = make_format_ctx_dataset(10, seed=0, seat=SELLER)
        for b, p in zip(bargain, probe):
            assert p["prompt"] == b["prompt"]
            assert (p["info"]["c"], p["info"]["v"]) == (b["info"]["c"], b["info"]["v"])

    def test_full_draw_distribution_no_zopa_kept(self):
        probe = make_format_ctx_dataset(200, seed=0)
        assert any(r["info"]["v"] <= r["info"]["c"] for r in probe)

    def test_scored_validity_only_accept_invalid_on_opening(self):
        info = make_format_ctx_dataset(1, seed=0)[0]["info"]
        assert format_reward(completion("#### OFFER(90)"), info) == 1.0
        assert format_reward(completion("#### WALK"), info) == 1.0
        assert format_reward(completion("#### ACCEPT"), info) == 0.0
        assert format_reward(completion("I offer 90"), info) == 0.0


class TestRetryProbe:
    def test_context_ends_with_env_retry_message_verbatim(self):
        row = make_retry_dataset(3, seed=0)[0]
        assert row["prompt"][-1] == {"role": "user", "content": RETRY_MESSAGE}
        assert row["prompt"][-2]["role"] == "assistant"

    def test_all_canonical_first_attempts_are_actually_invalid(self):
        for row in make_retry_dataset(6, seed=0):
            attempt = row["prompt"][-2]["content"]
            info = row["info"]
            assert parse_action(attempt, None, info["price_min"], info["price_max"]) is None

    def test_kinds_cycle_and_recorded(self):
        kinds = [r["info"]["invalid_kind"] for r in make_retry_dataset(6, seed=0)]
        assert kinds == ["prose_no_action", "missing_sentinel", "two_action_lines"] * 2

    def test_retry_scoring_matches_opening_turn_rules(self):
        info = make_retry_dataset(1, seed=0)[0]["info"]
        assert format_reward(completion("#### OFFER(95)"), info) == 1.0

        assert format_reward(completion("#### ACCEPT"), info) == 0.0


def _drive_opening(lines):
    env = OpeningProbeEnv(dataset=make_format_ctx_dataset(1, seed=0))
    row = env.dataset[0]
    state = vf.State.for_task({"prompt": row["prompt"], "info": row["info"], "answer": ""})

    async def go():
        messages = list(state["prompt"])
        responses = []
        for text in lines:
            if state.get("final_env_response") is not None:
                break
            messages.append(vf.AssistantMessage(content=text))
            resp = await env.env_response(messages, state)
            messages.extend(resp)
            responses.append(resp)
        return responses

    return state, asyncio.run(go())


class TestOpeningProbe:
    def test_valid_first_attempt_survives_immediately(self):
        state, responses = _drive_opening(["#### OFFER(95)"])
        assert opening_survival_reward(state) == 1.0
        assert state.get("final_env_response") is not None
        assert len(responses) == 1

    def test_invalid_then_valid_survives_via_real_retry_message(self):
        state, responses = _drive_opening(["let me think about pricing", "#### WALK"])
        assert responses[0][0].content == RETRY_MESSAGE
        assert opening_survival_reward(state) == 1.0

    def test_invalid_twice_forfeits(self):
        state, _ = _drive_opening(["no action", "still no action"])
        assert opening_survival_reward(state) == 0.0
        assert state.get("final_env_response") is not None

    def test_accept_lure_on_retry_forfeits_like_real_env(self):
        state, _ = _drive_opening(["no action", "#### ACCEPT"])
        assert opening_survival_reward(state) == 0.0

    def test_dataset_is_real_opening_context(self):
        env = OpeningProbeEnv(dataset=make_format_ctx_dataset(5, seed=0))
        bargain = make_dataset(5, seed=0, opponent="boulware", seat=SELLER)
        assert env.dataset[3]["prompt"] == bargain[3]["prompt"]

    def test_opening_aware_retry_style_uses_opening_message(self):
        from market.vf_env import RETRY_MESSAGE_OPENING

        env = OpeningProbeEnv(dataset=make_format_ctx_dataset(1, seed=0), retry_style="opening_aware")
        row = env.dataset[0]
        state = vf.State.for_task({"prompt": row["prompt"], "info": row["info"], "answer": ""})
        messages = list(state["prompt"]) + [vf.AssistantMessage(content="no action")]
        resp = asyncio.run(env.env_response(messages, state))
        assert resp[0].content == RETRY_MESSAGE_OPENING
        assert "ACCEPT" not in resp[0].content.split("not available")[1]
        with pytest.raises(ValueError, match="unknown retry_style"):
            OpeningProbeEnv(dataset=make_format_ctx_dataset(1, seed=0), retry_style="x")

    @pytest.mark.parametrize("seat", [SELLER, BUYER])
    @pytest.mark.parametrize(
        "variant",
        [
            {},
            {"scoring": "rolelocal"},
            {"termination": "reserve"},
            {"scoring": "rolelocal", "termination": "reserve"},
        ],
    )
    def test_variant_prompts_match_real_variant_env(self, seat, variant):


        env = load_environment(n=3, seed=0, task="opening", seat=seat, **variant)
        bargain = make_dataset(3, seed=0, opponent="boulware", seat=seat, **variant)
        for p, b in zip(env.dataset, bargain):
            assert p["prompt"] == b["prompt"]

    def test_variant_guards_on_other_tasks(self):
        with pytest.raises(ValueError, match="no scoring/termination"):
            load_probe("fullinfo", n=1, scoring="rolelocal")
        with pytest.raises(ValueError, match="no retry_style"):
            load_probe("format_ctx", n=1, retry_style="opening_aware")
        env = load_probe("opening", n=1, scoring="rolelocal", termination="reserve",
                         retry_style="opening_aware")
        assert isinstance(env, OpeningProbeEnv) and env.retry_style == "opening_aware"


class TestLoadEnvironmentDispatch:
    def test_probe_tasks_return_single_turn_envs(self):
        env = load_environment(n=4, task="format")
        assert isinstance(env, vf.SingleTurnEnv)
        env = load_environment(n=4, task="fullinfo", seat=BUYER)
        assert isinstance(env, vf.SingleTurnEnv)
        assert len(env.dataset) == 4

    def test_probe_rejects_opponent_args(self):
        with pytest.raises(ValueError, match="no opponent"):
            load_environment(n=4, task="format", opponent="mix")
        with pytest.raises(ValueError, match="no opponent"):
            load_environment(n=4, task="fullinfo", mix_weights={"conceder": 1.0})

    def test_unknown_task_and_unknown_kwarg_raise(self):
        with pytest.raises(ValueError, match="unknown probe task"):
            load_environment(n=4, task="anchoring")
        with pytest.raises(TypeError):
            load_environment(n=4, task="fullinfo", max_turns=3)

    def test_new_probe_tasks_dispatch(self):
        assert isinstance(load_environment(n=4, task="format_ctx"), vf.SingleTurnEnv)
        assert isinstance(load_environment(n=4, task="retry", seat=BUYER), vf.SingleTurnEnv)
        assert isinstance(load_environment(n=4, task="opening", seat=BUYER), OpeningProbeEnv)
        env = load_environment(n=4, task="fullinfo", formula="rolelocal", style="first")
        assert isinstance(env, vf.SingleTurnEnv)

    def test_formula_style_rejected_outside_fullinfo(self):
        with pytest.raises(ValueError, match="no formula/style"):
            load_environment(n=4, task="format_ctx", formula="rolelocal")
        with pytest.raises(ValueError, match="no formula/style"):
            load_environment(n=4, task="retry", style="first")

    def test_default_task_still_bargaining(self):
        env = load_environment(n=4, opponent="conceder")
        assert not isinstance(env, vf.SingleTurnEnv)


class TestWideDrawKwargs:


    WIDE = dict(c_lo=40, c_hi=60, v_lo=90, v_hi=110, t_lo=2, t_hi=6)

    def test_wide_draws_in_bounds_and_match_bargain_stream(self):
        ds = make_format_ctx_dataset(8, seed=999, seat=SELLER, **self.WIDE)
        bargain = make_dataset(8, seed=999, opponent="mix", seat=SELLER, **self.WIDE)
        for row, brow in zip(ds, bargain):
            assert row["info"]["c"] == brow["info"]["c"]
            assert row["info"]["v"] == brow["info"]["v"]
            assert 40 <= row["info"]["c"] <= 60
            assert 90 <= row["info"]["v"] <= 110

    def test_wide_prompt_carries_wide_priors_and_horizon(self):
        ds = make_format_ctx_dataset(4, seed=999, seat=BUYER, **self.WIDE)
        bargain = make_dataset(4, seed=999, opponent="mix", seat=BUYER, **self.WIDE)
        for row, brow in zip(ds, bargain):

            assert row["prompt"][0]["content"] == brow["prompt"][0]["content"]


    GOLDEN = {
        (0, SELLER): "e3305c56f3d5fb9a61f4135303a73148e4d608110f1f238c032187084be84006",
        (999, BUYER): "bddd0a87c48b76a24a49d2fbeafa7a3879d3ef488e5b9be91e0c2624a6cf5b54",
    }

    def test_defaults_byte_identical_to_prechange_golden(self):
        import hashlib
        import json

        for (seed, seat), expected in self.GOLDEN.items():
            ds = make_format_ctx_dataset(64, seed=seed, seat=seat)
            blob = json.dumps(
                [{"prompt": r["prompt"], "info": r["info"]} for r in ds], sort_keys=True
            )
            assert hashlib.sha256(blob.encode()).hexdigest() == expected

    def test_draw_bounds_validated_like_make_dataset(self):

        with pytest.raises(ValueError, match="together"):
            make_format_ctx_dataset(4, t_lo=2)
        with pytest.raises(ValueError, match="bad t range"):
            make_format_ctx_dataset(4, t_lo=1, t_hi=6)
        with pytest.raises(ValueError, match="bad draw bounds"):
            make_format_ctx_dataset(4, c_lo=80, c_hi=40)
        with pytest.raises(ValueError, match="c_lo <= v_lo"):
            make_format_ctx_dataset(4, c_lo=60, c_hi=80, v_lo=50, v_hi=100)
        with pytest.raises(ValueError, match="price grid"):
            make_format_ctx_dataset(4, v_lo=100, v_hi=200)

    def test_load_probe_opening_wide_passthrough(self):
        env = load_probe("opening", n=4, seed=999, seat=BUYER, **self.WIDE)
        assert isinstance(env, OpeningProbeEnv)
        for row in env.dataset:
            assert 40 <= row["info"]["c"] <= 60
            assert 90 <= row["info"]["v"] <= 110

    def test_load_probe_rejects_draw_kwargs_off_ctx_tasks(self):
        for task in ("format", "fullinfo", "retry"):
            with pytest.raises(ValueError, match="draw-bound"):
                load_probe(task, n=4, **self.WIDE)

    def test_load_environment_forwards_wide_opening(self):
        env = load_environment(n=4, seed=999, task="opening", seat=BUYER, **self.WIDE)
        assert isinstance(env, OpeningProbeEnv)
        for row in env.dataset:
            assert 40 <= row["info"]["c"] <= 60

    def test_load_environment_still_rejects_wide_fullinfo(self):
        with pytest.raises(ValueError, match="draw-bound"):
            load_environment(n=4, task="fullinfo", **self.WIDE)
