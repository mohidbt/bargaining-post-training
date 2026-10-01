

import random

from market.blind_search import (
    Camper,
    ConcessionBlind,
    evaluate,
    make_candidates,
    make_episodes,
    play_episode,
    search,
)
from market.bots import MidpointCamper
from market.env import BUYER, SELLER, Draw, Env
from market.vf_env import E_RANGES


def _env(draw, rnd=1, last_offer=None, transcript=()):
    env = Env(draw, random.Random(0), first_mover=SELLER)
    env.round = rnd
    env.last_offer = last_offer
    env.transcript = list(transcript)
    return env


DRAW_T = Draw(c=55, v=90, horizon=("T", 12))
DRAW_G = Draw(c=55, v=90, horizon=("geom", 0.95))


def test_concession_blind_ignores_transcript():
    for draw in (DRAW_T, DRAW_G):
        for rnd in range(1, 10):
            pol = ConcessionBlind(SELLER, draw.c, e=0.5, anchor_off=10, margin=0, floor_frac=0.5)
            bare = _env(draw, rnd)
            noisy = _env(
                draw,
                rnd,
                last_offer=(BUYER, 10),
                transcript=[(BUYER, "#### OFFER(10)"), (SELLER, "#### OFFER(140)")] * rnd,
            )
            assert pol.act(bare) == pol.act(noisy)
            assert pol.act(bare).startswith("#### OFFER(")


def test_concession_blind_buyer_side_blind():
    pol = ConcessionBlind(BUYER, DRAW_T.v, e=2.0, anchor_off=0, margin=0, floor_frac=1.0)
    bare = _env(DRAW_T, 3)
    noisy = _env(DRAW_T, 3, last_offer=(SELLER, 150), transcript=[(SELLER, "#### OFFER(150)")])
    assert pol.act(bare) == pol.act(noisy)


def test_camper_ignores_transcript_and_matches_midpoint_camper():
    camp = Camper(SELLER, DRAW_T.c, price=70)
    ref = MidpointCamper(SELLER, 70)
    for last in (None, (BUYER, 10), (BUYER, 69)):
        env = _env(DRAW_T, 4, last_offer=last)
        assert camp.act(env) == ref.act(env)
    assert camp.act(_env(DRAW_T, 4, last_offer=(BUYER, 71))) == "#### ACCEPT"


def test_camper_offset_uses_own_value():
    assert Camper(SELLER, 60, offset=15).price == 75
    assert Camper(BUYER, 90, offset=15).price == 75
    assert Camper(SELLER, 60, price=200).price == 150


def test_episodes_match_training_distribution():


    eps = make_episodes(300, SELLER, seed=7)
    assert eps == make_episodes(300, SELLER, seed=7)
    for ep in eps:
        assert 40 <= ep.draw.c <= 80 and 50 <= ep.draw.v <= 100
        if ep.opp_name == "random_threshold":
            assert ep.opp_e is None
            assert 40 <= ep.opp_anchor <= ep.draw.v
        else:
            lo, hi = E_RANGES[ep.opp_name]
            assert lo <= ep.opp_e <= hi
            assert 10 <= ep.opp_anchor <= 50
    names = {ep.opp_name for ep in eps}
    assert names == {"boulware", "conceder", "random_threshold"}
    for ep in make_episodes(80, BUYER, seed=7):
        if ep.opp_name == "random_threshold":
            assert ep.draw.c <= ep.opp_anchor <= 100
        else:
            assert 90 <= ep.opp_anchor <= 130


def test_opponent_pool_matches_vf_env():
    from market.blind_search import OPPONENTS
    from market.vf_env import BOT_FACTORIES

    assert set(OPPONENTS) == set(BOT_FACTORIES) == {"boulware", "conceder", "random_threshold"}


def test_make_episodes_mode_filter():

    for mode in ("T", "geom"):
        eps = make_episodes(80, SELLER, seed=5, mode=mode)
        assert len(eps) == 80
        assert all(ep.draw.horizon[0] == mode for ep in eps)
        assert eps == make_episodes(80, SELLER, seed=5, mode=mode)


def test_play_episode_zero_sum():
    factory = lambda role, own: ConcessionBlind(role, own, e=1.0, anchor_off=0, margin=0, floor_frac=0.5)
    for seat in (SELLER, BUYER):
        for ep in make_episodes(100, seat, seed=3):
            r_me, r_opp, outcome = play_episode(factory, seat, ep)
            assert r_me + r_opp == 0
            assert -1.0 <= r_me <= 1.0
            assert outcome[0] in {"deal", "walk", "timeout", "breakdown", "forfeit"}


def test_evaluate_bounds():
    mean, deal_rate = evaluate(lambda role, own: Camper(role, own, price=70), SELLER, make_episodes(60, SELLER, 1))
    assert -1.0 <= mean <= 1.0
    assert 0.0 <= deal_rate <= 1.0


def test_search_deterministic_under_seed():
    cands = make_candidates()[:6]
    a = search(n_screen=25, n_refine=25, seed=11, top_k=3, candidates=cands)
    b = search(n_screen=25, n_refine=25, seed=11, top_k=3, candidates=cands)
    assert a == b
    assert len(a["blind"]) == 3
    assert "oracle" in a["baselines"]

    means = [r["mean"] for r in a["blind"]]
    assert means == sorted(means, reverse=True)
