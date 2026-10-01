import random

import pytest

from market.env import BUYER, SELLER, Draw, Env, sample_draw

ZOPA = Draw(c=40, v=100, horizon=("T", 10))


def env(draw=ZOPA, **kw):
    return Env(draw, random.Random(0), **kw)


def test_deal_at_midpoint_is_zero_zero():
    e = env()
    e.step("#### OFFER(70)")
    e.step("#### ACCEPT")
    assert e.outcome == ("deal", 70)
    assert e.rewards() == (0.0, 0.0)


def test_reward_formula_and_zero_sum():
    e = env()
    e.step("<think>anchor</think> #### OFFER(85)")
    e.step("#### ACCEPT")
    r_s, r_b = e.rewards()
    assert r_s == pytest.approx((2 * 85 - 100 - 40) / 60)
    assert r_s + r_b == 0


def test_accept_binds_at_last_offer():
    e = env()
    e.step("#### OFFER(90)")
    e.step("#### OFFER(50)")
    e.step("#### ACCEPT")
    assert e.outcome == ("deal", 50)


def test_walk_is_zero_zero():
    e = env()
    e.step("#### OFFER(90)")
    e.step("#### WALK")
    assert e.rewards() == (0.0, 0.0)


def test_invalid_gets_one_retry_then_forfeit():
    e = env()
    assert e.step("hello") is None and not e.done
    assert e.turn == SELLER
    e.step("still not an action")
    assert e.outcome == ("forfeit", SELLER)
    assert e.rewards() == (-1.0, 1.0)


def test_accept_with_no_standing_offer_is_invalid():
    e = env()
    e.step("#### ACCEPT")
    e.step("#### ACCEPT")
    assert e.outcome == ("forfeit", SELLER)


def test_offer_off_grid_is_invalid():
    e = env()
    e.step("#### OFFER(9999)")
    e.step("#### OFFER(-5)")
    assert e.outcome == ("forfeit", SELLER)


def test_valid_after_retry_continues():
    e = env()
    e.step("garbage")
    e.step("#### OFFER(80)")
    assert not e.done and e.turn == BUYER


def test_timeout_after_t_rounds():
    e = env(Draw(c=40, v=100, horizon=("T", 2)))
    e.step("#### OFFER(90)")
    e.step("#### OFFER(50)")
    assert e.outcome == ("timeout",)
    assert e.rewards() == (0.0, 0.0)


def test_ultimatum_accept_at_t2():
    e = env(Draw(c=40, v=100, horizon=("T", 2)))
    e.step("#### OFFER(90)")
    e.step("#### ACCEPT")
    assert e.outcome == ("deal", 90)


def test_geometric_breakdown():
    e = env(Draw(c=40, v=100, horizon=("geom", 0.0)))
    e.step("#### OFFER(90)")
    assert e.outcome == ("breakdown",)
    assert e.rewards() == (0.0, 0.0)


def test_no_zopa_walk_is_zero_deal_is_plus_minus_one():
    no_zopa = Draw(c=80, v=50, horizon=("T", 10))
    e = env(no_zopa)
    e.step("#### OFFER(90)")
    e.step("#### WALK")
    assert e.rewards() == (0.0, 0.0)

    e = env(no_zopa)
    e.step("#### OFFER(90)")
    e.step("#### ACCEPT")
    assert e.rewards() == (1.0, -1.0)

    e = env(no_zopa)
    e.step("#### OFFER(90)")
    e.step("#### OFFER(55)")
    e.step("#### ACCEPT")
    assert e.rewards() == (-1.0, 1.0)


def test_zero_width_draw_deal_at_value_is_zero_zero():
    e = env(Draw(c=60, v=60, horizon=("T", 10)))
    e.step("#### OFFER(60)")
    e.step("#### ACCEPT")
    assert e.rewards() == (0.0, 0.0)


def test_deal_outside_zopa_is_clamped():
    e = env()
    e.step("#### OFFER(120)")
    e.step("#### ACCEPT")
    assert e.rewards() == (1.0, -1.0)


def test_zero_sum_invariant_random_playouts():
    rng = random.Random(1)
    actions = ["#### OFFER({p})", "#### ACCEPT", "#### WALK", "garbage"]
    for _ in range(500):
        e = Env(sample_draw(rng), random.Random(rng.getrandbits(32)))
        while not e.done:
            a = rng.choice(actions).format(p=rng.randint(-10, 200))
            e.step(a)
        r_s, r_b = e.rewards()
        assert r_s + r_b == 0
        assert -1 <= r_s <= 1


def test_action_inside_think_is_ignored():
    e = env()
    e.step("<think>maybe #### OFFER(150)</think>\n#### WALK")
    assert e.outcome == ("walk", SELLER)


def test_invalid_action_inside_think_does_not_burn_retry():
    e = env()
    e.step("<think>bad #### OFFER(9999)</think>\n#### WALK")
    assert e.outcome == ("walk", SELLER)


def test_uppercase_think_block_is_stripped_too():
    e = env()

    e.step("<THINK>\n#### OFFER(150)\n</THINK>\n#### WALK")
    assert e.outcome == ("walk", SELLER)


def test_unclosed_uppercase_think_hides_action():
    e = env()
    e.step("<THINK>rambling\n#### OFFER(150)")
    assert not e.done and e.retry_used


def test_unclosed_think_hides_action():
    e = env()
    e.step("<think>rambling #### OFFER(150)")
    assert not e.done and e.retry_used


def test_multiple_action_lines_are_invalid():
    e = env()
    e.step("#### OFFER(90)\n#### WALK")
    assert not e.done and e.retry_used
    e.step("#### OFFER(90)")
    assert e.last_offer == (SELLER, 90)


def test_action_must_be_line_anchored():
    e = env()
    e.step("I will not say #### ACCEPT yet, thinking...")
    assert not e.done and e.retry_used
