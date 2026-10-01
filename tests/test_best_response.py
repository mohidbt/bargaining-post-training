

import random

import pytest

from market import bots
from market.best_response import BestResponse, expected_value
from market.blind_search import ConcessionBlind
from market.env import BUYER, SELLER, Draw, Env
from market.env import sample_draw
from market.vf_env import sample_opponent_params


def test_ultimatum_vs_random_threshold_seller():


    draw = Draw(c=50, v=90, horizon=("T", 2))
    br = BestResponse(draw, SELLER, bots.RandomThreshold(BUYER, 90, 80))
    assert br.value == 0.5
    env = Env(draw, random.Random(0), first_mover=SELLER)
    assert br.act(env) == "#### OFFER(80)"


def test_ultimatum_unprofitable_threshold_scores_zero():


    draw = Draw(c=50, v=90, horizon=("T", 2))
    br = BestResponse(draw, SELLER, bots.RandomThreshold(BUYER, 90, 45))
    assert br.value == 0.0


def test_ultimatum_buyer_seat():


    draw = Draw(c=50, v=90, horizon=("T", 2))
    br = BestResponse(draw, BUYER, bots.RandomThreshold(SELLER, 50, 60))
    assert br.value == 0.5
    env = Env(draw, random.Random(0), first_mover=BUYER)
    assert br.act(env) == "#### OFFER(60)"


def test_deadline_extraction_vs_timebot_t4():


    draw = Draw(c=50, v=90, horizon=("T", 4))
    br = BestResponse(draw, SELLER, bots.TimeBot(BUYER, 90, e=1.0, anchor=40))
    assert br.value == 1.0


def test_geom_vs_random_threshold_closed_form():


    draw = Draw(c=50, v=90, horizon=("geom", 0.9))
    br = BestResponse(draw, SELLER, bots.RandomThreshold(BUYER, 90, 80))
    assert br.value == pytest.approx(0.45, abs=1e-12)


def test_no_zopa_dp_walks_to_zero():


    draw = Draw(c=60, v=45, horizon=("T", 6))
    br = BestResponse(draw, SELLER, bots.TimeBot(BUYER, 45, e=1.0, anchor=40))
    assert br.value == 0.0


class _MultiActionBot(bots.RandomThreshold):

    def act(self, env):
        return super().act(env) + "\n#### WALK"


class _OffGridBot(bots.RandomThreshold):

    def act(self, env):
        return "#### OFFER(999)"


def test_malformed_bot_text_is_handled_exactly_as_the_env_would():


    draw = Draw(c=50, v=90, horizon=("T", 2))
    for bot in (_MultiActionBot(BUYER, 90, 80), _OffGridBot(BUYER, 90, 80)):
        br = BestResponse(draw, SELLER, bot)
        assert br.value == 1.0
        ev, outcome = expected_value(br, SELLER, draw, bot)
        assert ev == br.value
        assert outcome == ("forfeit", BUYER)


    geom = Draw(c=50, v=90, horizon=("geom", 0.9))
    br = BestResponse(geom, SELLER, _OffGridBot(BUYER, 90, 80))
    assert br.value == pytest.approx(0.9, abs=1e-12)
    ev, outcome = expected_value(br, SELLER, geom, _OffGridBot(BUYER, 90, 80))
    assert ev == pytest.approx(br.value, abs=1e-12)
    assert outcome == ("forfeit", BUYER)


def test_titfortat_out_of_scope():
    draw = Draw(c=50, v=90, horizon=("T", 4))
    with pytest.raises(ValueError):
        BestResponse(draw, SELLER, bots.TitForTat(BUYER, 90, anchor=40))


def _random_episode(rng):

    seat = rng.choice((SELLER, BUYER))
    opp_role = BUYER if seat == SELLER else SELLER
    draw = sample_draw(rng)
    name = rng.choice(("boulware", "conceder", "random_threshold"))
    opp_own = draw.c if opp_role == SELLER else draw.v
    e, anchor = sample_opponent_params(rng, name, opp_role, 0, 150, opp_own)
    bot = getattr(bots, name)(opp_role, opp_own, anchor=anchor, e=e)
    return draw, seat, bot


def test_dp_upper_bounds_all_simulated_policies():


    rng = random.Random(7)
    for _ in range(50):
        draw, seat, bot = _random_episode(rng)
        br = BestResponse(draw, seat, bot)
        assert 0.0 <= br.value <= 1.0
        own = draw.c if seat == SELLER else draw.v
        policies = (
            bots.OracleBot(seat),
            bots.boulware(seat, own, anchor=100 if seat == SELLER else 40),
            ConcessionBlind(seat, own, e=0.1, anchor_off=0, margin=0, floor_frac=0.5),
        )
        for policy in policies:
            ev, _ = expected_value(policy, seat, draw, bot)
            assert ev <= br.value + 1e-9


def test_dp_policy_replay_through_real_env_hits_dp_value():


    rng = random.Random(11)
    for _ in range(40):
        draw, seat, bot = _random_episode(rng)
        br = BestResponse(draw, seat, bot)
        ev, _ = expected_value(br, seat, draw, bot)
        assert ev == pytest.approx(br.value, abs=1e-9)


def test_replay_terminates_and_is_valid_in_real_env():


    rng = random.Random(13)
    for _ in range(20):
        draw, seat, bot = _random_episode(rng)
        br = BestResponse(draw, seat, bot)
        env = Env(draw, random.Random(3), first_mover=seat)
        actors = {seat: br, (BUYER if seat == SELLER else SELLER): bot}
        for _step in range(500):
            if env.done:
                break
            env.step(actors[env.turn].act(env))
        assert env.done
        assert env.outcome[0] != "forfeit"
