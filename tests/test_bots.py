import random

from market.bots import RandomThreshold, TitForTat, boulware, conceder
from market.env import BUYER, SELLER, Draw, Env, play
from market.tournament import run

ZOPA = Draw(c=40, v=100, horizon=("T", 12))


def test_boulware_concedes_later_than_conceder():
    hard = boulware(SELLER, 40, anchor=100)
    soft = conceder(SELLER, 40, anchor=100)
    e = Env(ZOPA, random.Random(0))
    e.round = 6
    assert hard.target(e) > soft.target(e)


def test_bots_never_offer_past_reservation():
    for role, own, anchor in ((SELLER, 60, 100), (BUYER, 70, 40)):
        bot = conceder(role, own, anchor=anchor)
        e = Env(ZOPA, random.Random(0))
        for rnd in range(1, 13):
            e.round = rnd
            t = bot.target(e)
            assert t >= own if role == SELLER else t <= own


def test_scripted_pair_reaches_deal_on_zopa():
    e = Env(ZOPA, random.Random(0))
    r_s, r_b = play(e, boulware(SELLER, ZOPA.c, 100), conceder(BUYER, ZOPA.v, 40))
    assert e.outcome[0] == "deal"
    assert r_s + r_b == 0


def test_scripted_pair_no_deal_on_no_zopa():
    draw = Draw(c=90, v=45, horizon=("T", 12))
    e = Env(draw, random.Random(0))
    play(e, boulware(SELLER, draw.c, 100), boulware(BUYER, draw.v, 40))
    assert e.outcome[0] != "deal"


def test_titfortat_plays_valid_episode():
    e = Env(ZOPA, random.Random(0))
    r_s, r_b = play(e, TitForTat(SELLER, ZOPA.c, 100), TitForTat(BUYER, ZOPA.v, 40))
    assert e.outcome[0] != "forfeit"
    assert r_s + r_b == 0


def test_tournament_boulware_beats_conceder():
    reward, _, deal_rate, nodeal_rate = run(n_games=300, seed=0)
    mean = {k: sum(v) / len(v) for k, v in reward.items()}
    assert mean["boulware"] > mean["conceder"]


    assert deal_rate > 0.4
    assert nodeal_rate == 1.0


def test_midpoint_camper_camps_and_accepts_only_at_or_better_than_mhat():
    from market.bots import MidpointCamper

    seller = MidpointCamper(SELLER, m_hat=68)
    e = Env(ZOPA, random.Random(0))
    assert seller.act(e) == "#### OFFER(68)"
    e.step("#### OFFER(70)")
    e.step("#### OFFER(69)")
    assert seller.act(e) == "#### ACCEPT"
    e2 = Env(ZOPA, random.Random(0))
    e2.step("#### OFFER(70)")
    e2.step("#### OFFER(60)")
    assert seller.act(e2) == "#### OFFER(68)"


def test_random_threshold_offers_threshold_and_accepts_at_or_above_it():
    seller = RandomThreshold(SELLER, own_value=50, threshold=70)
    e = Env(ZOPA, random.Random(0))
    assert seller.act(e) == "#### OFFER(70)"
    e.step("#### OFFER(70)")
    e.step("#### OFFER(69)")
    assert seller.act(e) == "#### OFFER(70)"
    e2 = Env(ZOPA, random.Random(0))
    e2.step("#### OFFER(70)")
    e2.step("#### OFFER(70)")
    assert RandomThreshold(SELLER, 50, 70).act(e2) == "#### ACCEPT"

    buyer = RandomThreshold(BUYER, own_value=90, threshold=60)
    e3 = Env(ZOPA, random.Random(0))
    e3.step("#### OFFER(61)")
    assert buyer.act(e3) == "#### OFFER(60)"
    e4 = Env(ZOPA, random.Random(0))
    e4.step("#### OFFER(59)")
    assert RandomThreshold(BUYER, 90, 60).act(e4) == "#### ACCEPT"


def test_random_threshold_never_trades_at_a_loss():

    seller = RandomThreshold(SELLER, own_value=60, threshold=50)
    assert seller.threshold == 60
    e = Env(ZOPA, random.Random(0))
    e.step("#### OFFER(120)")
    e.step("#### OFFER(59)")
    assert seller.act(e) == "#### OFFER(60)"
    buyer = RandomThreshold(BUYER, own_value=70, threshold=80)
    assert buyer.threshold == 70


def test_random_threshold_is_transcript_blind():
    bot = RandomThreshold(SELLER, 50, 70)
    bare = Env(ZOPA, random.Random(0))
    bare.last_offer = (BUYER, 10)
    noisy = Env(ZOPA, random.Random(0))
    noisy.round = 9
    noisy.last_offer = (BUYER, 10)
    noisy.transcript = [(BUYER, "#### OFFER(10)"), (SELLER, "#### OFFER(140)")] * 4
    assert bot.act(bare) == bot.act(noisy) == "#### OFFER(70)"


def test_oracle_bot_extracts_more_than_any_blind_bot():


    from market.bots import OracleBot

    draws = [Draw(c=40, v=100, horizon=("T", 12)), Draw(c=55, v=95, horizon=("T", 8))]
    for draw in draws:
        e = Env(draw, random.Random(0))
        r_oracle, _ = play(e, OracleBot(SELLER), conceder(BUYER, draw.v, 40))
        e2 = Env(draw, random.Random(0))
        r_blind, _ = play(e2, boulware(SELLER, draw.c, 100), conceder(BUYER, draw.v, 40))
        assert e.outcome[0] == "deal"
        assert r_oracle >= r_blind
        assert r_oracle > 0


def test_oracle_never_accepts_below_true_midpoint():
    from market.bots import OracleBot

    draw = Draw(c=40, v=100, horizon=("T", 3))
    e = Env(draw, random.Random(0))
    e.step("#### OFFER(120)")
    e.step("#### OFFER(69)")
    assert OracleBot(SELLER).act(e) == "#### WALK"
    e2 = Env(draw, random.Random(0))
    e2.step("#### OFFER(120)")
    e2.step("#### OFFER(71)")
    assert OracleBot(SELLER).act(e2) == "#### ACCEPT"


def test_oracle_accepts_reasonable_offer_late_in_geometric_horizon():
    from market.bots import OracleBot

    draw = Draw(c=40, v=100, horizon=("geom", 0.85))
    e = Env(draw, random.Random(42))
    e.round = 8
    e.last_offer = (BUYER, 75)
    assert OracleBot(SELLER).act(e) == "#### ACCEPT"


def test_oracle_accepts_midpoint_beating_offer_early_under_breakdown_risk():
    from market.bots import OracleBot

    draw = Draw(c=40, v=100, horizon=("geom", 0.85))
    e = Env(draw, random.Random(0))
    e.step("#### OFFER(100)")
    e.step("#### OFFER(80)")
    assert OracleBot(SELLER).act(e) == "#### ACCEPT"
