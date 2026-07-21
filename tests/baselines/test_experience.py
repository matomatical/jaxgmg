"""
Tests for the return- and advantage-estimation math in
``jaxgmg.baselines.experience``:

* ``generalised_advantage_estimation`` (GAE) — the advantage signal PPO trains
  on, and the basis of the ``pvl`` regret estimator;
* ``compute_average_return`` / ``compute_maximum_return`` — multi-episode-aware
  return summaries that underpin the default ``maxmc-actor`` regret estimator.

These are pure, training-free functions, so we check them with hand-computed
golden values, an independent NumPy reference, and structural properties
(λ=0 ⇒ TD error; ``done`` cuts the bootstrap; multi-episode masking).
"""

import numpy as np

import jax.numpy as jnp
import pytest

from jaxgmg.baselines import experience


# --- independent NumPy reference for GAE ---------------------------------- #

def reference_gae(rewards, values, dones, final_value, lambda_, discount_rate):
    """Plain reverse loop mirroring the documented GAE recursion."""
    n = len(rewards)
    out = np.zeros(n)
    gae = 0.0
    next_value = float(final_value)
    for t in reversed(range(n)):
        gae = (
            rewards[t] - values[t]
            + (1.0 - dones[t]) * discount_rate * (next_value + lambda_ * gae)
        )
        out[t] = gae
        next_value = values[t]
    return out


def run_gae(rewards, values, dones, final_value, lambda_, discount_rate):
    return np.asarray(experience.generalised_advantage_estimation(
        rewards=jnp.asarray(rewards, dtype=float),
        dones=jnp.asarray(dones, dtype=bool),
        values=jnp.asarray(values, dtype=float),
        final_value=jnp.asarray(final_value, dtype=float),
        lambda_=lambda_,
        discount_rate=discount_rate,
    ))


# --- GAE: golden ----------------------------------------------------------- #

def test_gae_golden_single_episode():
    # hand-computed (see comments) for a 3-step rollout with no terminations.
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.0, 1.5]
    final_value = 2.0
    gamma, lam = 0.9, 0.8
    # delta_t = r_t + gamma*next_value - v_t
    #   d2 = 3 + 0.9*2.0 - 1.5 = 3.3
    #   d1 = 2 + 0.9*1.5 - 1.0 = 2.35
    #   d0 = 1 + 0.9*1.0 - 0.5 = 1.4
    # gae2 = 3.3
    # gae1 = 2.35 + 0.9*0.8*3.3 = 4.726
    # gae0 = 1.4 + 0.9*0.8*4.726 = 4.80272
    got = run_gae(rewards, values, dones=[0, 0, 0],
                  final_value=final_value, lambda_=lam, discount_rate=gamma)
    np.testing.assert_allclose(got, [4.80272, 4.726, 3.3], rtol=1e-6)


def test_gae_matches_reference_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(2, 12))
        rewards = rng.normal(size=n)
        values = rng.normal(size=n)
        dones = rng.integers(0, 2, size=n).astype(float)
        final_value = float(rng.normal())
        gamma = float(rng.uniform(0.8, 0.999))
        lam = float(rng.uniform(0.0, 1.0))
        got = run_gae(rewards, values, dones, final_value, lam, gamma)
        ref = reference_gae(rewards, values, dones, final_value, lam, gamma)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


# --- GAE: properties ------------------------------------------------------- #

def test_gae_lambda_zero_is_td_error():
    # with lambda=0, GAE collapses to the one-step TD error
    #   delta_t = r_t + gamma*(1-done_t)*next_value_t - v_t
    # where next_value_t is values[t+1] (interior) or final_value (last step).
    rewards = np.array([1.0, -0.5, 2.0, 0.0])
    values = np.array([0.2, 0.4, 0.6, 0.8])
    dones = np.array([0.0, 1.0, 0.0, 0.0])
    final_value = 1.3
    gamma = 0.95
    got = run_gae(rewards, values, dones, final_value, lambda_=0.0,
                  discount_rate=gamma)
    next_values = np.append(values[1:], final_value)
    td = rewards + gamma * (1 - dones) * next_values - values
    np.testing.assert_allclose(got, td, rtol=1e-6)


def test_gae_done_cuts_the_bootstrap():
    # GAE at and before a terminal step must not depend on anything after it.
    base_r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    base_v = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    dones = np.array([0.0, 0.0, 1.0, 0.0, 0.0])   # episode ends at index 2
    gamma, lam = 0.9, 0.7

    g1 = run_gae(base_r, base_v, dones, final_value=0.0,
                 lambda_=lam, discount_rate=gamma)
    # perturb everything strictly after the terminal step
    r2 = base_r.copy(); r2[3:] += 10.0
    v2 = base_v.copy(); v2[3:] += 10.0
    g2 = run_gae(r2, v2, dones, final_value=99.0,
                 lambda_=lam, discount_rate=gamma)

    # advantages up to and including the terminal step are unchanged
    np.testing.assert_allclose(g1[:3], g2[:3], rtol=1e-6)
    # at the terminal step the advantage is exactly reward - value
    np.testing.assert_allclose(g1[2], base_r[2] - base_v[2], rtol=1e-6)


def test_gae_plus_value_is_discounted_return_when_lambda_one():
    # with lambda=1 and a single non-terminating episode, gae_t + value_t is
    # the Monte-Carlo discounted return-to-go bootstrapped by final_value.
    rewards = np.array([1.0, 0.0, 2.0, -1.0])
    values = np.array([0.3, 0.7, 0.1, 0.9])
    final_value = 0.5
    gamma = 0.9
    got = run_gae(rewards, values, dones=[0, 0, 0, 0],
                  final_value=final_value, lambda_=1.0, discount_rate=gamma)
    targets = got + values
    # direct discounted return-to-go with bootstrap
    n = len(rewards)
    expected = np.zeros(n)
    acc = final_value
    for t in reversed(range(n)):
        acc = rewards[t] + gamma * acc
        expected[t] = acc
    np.testing.assert_allclose(targets, expected, rtol=1e-6)


def test_batch_gae_matches_per_row():
    rng = np.random.default_rng(1)
    L, S = 5, 7
    rewards = rng.normal(size=(L, S))
    values = rng.normal(size=(L, S))
    dones = rng.integers(0, 2, size=(L, S)).astype(float)
    final_values = rng.normal(size=L)
    gamma, lam = 0.99, 0.95
    batched = np.asarray(experience.batch_generalised_advantage_estimation(
        rewards=jnp.asarray(rewards),
        dones=jnp.asarray(dones, dtype=bool),
        values=jnp.asarray(values),
        final_values=jnp.asarray(final_values),
        lambda_=lam,
        discount_rate=gamma,
    ))
    for i in range(L):
        ref = reference_gae(rewards[i], values[i], dones[i],
                            final_values[i], lam, gamma)
        np.testing.assert_allclose(batched[i], ref, rtol=1e-5, atol=1e-6)


# --- average / maximum return --------------------------------------------- #

def avg_return(rewards, dones, gamma):
    return float(experience.compute_average_return(
        rewards=jnp.asarray(rewards, dtype=float),
        dones=jnp.asarray(dones, dtype=bool),
        discount_rate=gamma,
    ))


def max_return(rewards, dones, gamma):
    return float(experience.compute_maximum_return(
        rewards=jnp.asarray(rewards, dtype=float),
        dones=jnp.asarray(dones, dtype=bool),
        discount_rate=gamma,
    ))


def test_single_episode_return_is_discounted_sum():
    # one episode: return at the start is sum of gamma^t * reward_t.
    rewards = [0.0, 0.0, 1.0]
    dones = [0, 0, 1]
    gamma = 0.5
    # 0 + 0 + 0.5^2 * 1 = 0.25
    assert avg_return(rewards, dones, gamma) == pytest.approx(0.25)
    assert max_return(rewards, dones, gamma) == pytest.approx(0.25)


def test_two_episode_average_and_maximum():
    # episode A: rewards [0,0,1] done at step2 -> return 0.25 (gamma=0.5)
    # episode B: rewards [0,1]   done at step4 -> return 0.5
    rewards = [0.0, 0.0, 1.0, 0.0, 1.0]
    dones = [0, 0, 1, 0, 1]
    gamma = 0.5
    assert avg_return(rewards, dones, gamma) == pytest.approx((0.25 + 0.5) / 2)
    assert max_return(rewards, dones, gamma) == pytest.approx(0.5)


def test_average_return_ignores_partial_trailing_episode_correctly():
    # the multi-episode masking keys off `first_steps = roll(dones,1)[0]=True`.
    # With a single episode that never terminates, the whole rollout is one
    # episode starting at step 0; the average equals that single return.
    rewards = [1.0, 1.0, 1.0]
    dones = [0, 0, 0]
    gamma = 0.9
    expected = 1.0 + 0.9 * 1.0 + 0.81 * 1.0   # 2.71
    assert avg_return(rewards, dones, gamma) == pytest.approx(expected)


def test_first_step_masking_counts_episodes():
    # three single-step episodes (done every step): average is the mean reward,
    # maximum is the largest reward.
    rewards = [2.0, 5.0, 3.0]
    dones = [1, 1, 1]
    gamma = 0.9
    assert avg_return(rewards, dones, gamma) == pytest.approx((2 + 5 + 3) / 3)
    assert max_return(rewards, dones, gamma) == pytest.approx(5.0)
