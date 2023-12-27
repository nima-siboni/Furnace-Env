# pylint: disable=protected-access
"""Test the furnace environment."""
from __future__ import annotations

from inspect import signature

import gymnasium as gym
import numpy as np

from furnace import Furnace
from utils import FurnaceConfig


def test_init():
    """Test initializing the environment."""
    env = Furnace()
    assert env is not None
    assert isinstance(env, Furnace)
    config = env.cfg
    assert isinstance(config, FurnaceConfig)


def test_action_space():
    """Test the action space."""
    env = Furnace()
    assert env._action_space is not None
    assert env._action_space.n == 4

    config = env.cfg
    config.use_stop_action = False
    env = Furnace(dict(config))
    assert env._action_space.n == 3


def test_observation_space():
    """Test the observation space."""
    env = Furnace()
    assert env._observation_space is not None

    assert isinstance(env._observation_space, gym.spaces.Dict)

    assert len(env._observation_space.spaces) == 3

    assert env._observation_space['PF'].shape == (
        env.cfg.dimension, env.cfg.dimension, 1,
    )
    assert env._observation_space['temperature'].shape == (1,)
    assert env._observation_space['timestep'].shape == (1,)


def test_reset():
    """Test resetting the environment."""
    env = Furnace()
    obs, _ = env.reset()
    assert obs is not None
    assert isinstance(obs, dict)
    assert 'PF' in obs, 'The observation should contain a PF.'
    assert 'temperature' in obs, 'The observation should contain a temperature.'
    assert 'timestep' in obs, 'The observation should contain a timestep.'

    assert obs['PF'].shape == (
        env.cfg.dimension, env.cfg.dimension, 1,
    ), 'The PF has the wrong shape.'
    assert len(obs['temperature']) == 1, 'The temperature should be a scalar.'
    assert len(obs['timestep']) == 1, 'The timestep should be a scalar.'

    assert obs['PF'].max() <= 1 + env.cfg.shift_pf, 'The PF is out of range.'
    assert obs['PF'].min() >= 0. + env.cfg.shift_pf, 'The PF is out of range.'
    assert obs['PF'].max() != obs['PF'].min(), 'The PF is constant.'

    assert obs['temperature'][0] <= 1., 'The temperature is larger than 1.0.'
    assert obs['temperature'][0] >= 0., 'The temperature is smaller than 0.0.'
    assert obs['timestep'][0] == 0., 'The timestep is not 0.'

    # Test that the reset function resets the steps counter.
    env = Furnace()
    env.reset()
    assert env.steps == 0, 'The steps counter should be 0.'

    # Test that the seed is effective.
    env = Furnace()
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    obs3, _ = env.reset(seed=43)
    assert np.array_equal(obs1['PF'], obs2['PF']), 'The PF should be the same.'
    assert not np.array_equal(
        obs1['PF'], obs3['PF'],
    ), 'The PF should be different.'


def test_step_signature():
    """Test that the step function has the correct signature."""
    env = Furnace()
    assert hasattr(env, 'step')
    assert callable(env.step)
    sig = signature(env.step)
    assert len(sig.parameters) == 1, 'The step function should take one argument.'
    assert 'action' in sig.parameters, 'The step function should take an action as argument.'
    assert sig.parameters['action'].annotation == 'int', 'The action should be an integer.'


def test_step_returned_tuple():
    """Test that the step function returns a tuple."""
    env = Furnace()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_temperature_terminated_truncated():
    """Test that the step function changes the temperature, terminated, truncated."""
    env = Furnace()
    # lets calculate the temperature change
    config = env.cfg
    delta_temperature = config.temperature_change_per_step / \
        (config.maximum_temperature - config.minimum_temperature)
    obs, _ = env.reset()

    # Action 0: decrease temperature
    initial_temperature = obs['temperature'][0]
    obs, _, terminated, truncated, _ = env.step(0)
    current_temperature = obs['temperature'][0]
    assert current_temperature == initial_temperature - \
        delta_temperature, 'The temperature should decrease.'
    assert terminated is False, 'The episode should not be terminated.'
    assert truncated is False, 'The episode should not be truncated.'
    # Action 1: do nothing
    initial_temperature = obs['temperature'][0]
    obs, _, terminated, truncated, _ = env.step(1)
    current_temperature = obs['temperature'][0]
    assert current_temperature == initial_temperature, 'The temperature should stay same.'
    assert terminated is False, 'The episode should not be terminated.'
    assert truncated is False, 'The episode should not be truncated.'
    # Action 2: increase temperature
    initial_temperature = obs['temperature'][0]
    obs, _, terminated, truncated, _ = env.step(2)
    current_temperature = obs['temperature'][0]
    assert current_temperature == initial_temperature + \
        delta_temperature, 'The temperature should increase.'
    assert terminated is False, 'The episode should not be terminated.'
    assert truncated is False, 'The episode should not be truncated.'

    # Action 3: Episode termination action
    # obs before the action
    original_obs = obs.copy()
    initial_temperature = obs['temperature'][0]
    obs, _, terminated, truncated, _ = env.step(3)
    assert terminated is True, 'The episode should be terminated.'
    assert truncated is False, 'The episode should not be truncated.'

    current_temperature = obs['temperature'][0]
    assert current_temperature == initial_temperature, \
        'The temperature should increase when the process is terminated.'

    assert np.array_equal(
        obs['PF'], original_obs['PF'],
    ), 'The PF should not change when the process is terminated.'
    delta_t = 1 / config.horizon
    assert obs['timestep'][0] == original_obs['timestep'][0] + delta_t


def test_temperature_range():
    """Test that the temperature is in the correct range."""
    env = Furnace()
    obs, _ = env.reset()
    # Action 0: decrease temperature
    while obs['temperature'][0] > 0.:
        obs, _, _, _, _ = env.step(0)
        assert obs['temperature'][0] >= 0., 'The temperature should not decrease below 0.'

    obs, _, _, _, _ = env.step(0)
    current_temperature = obs['temperature'][0]
    assert current_temperature == 0., 'The temperature should not decrease below 0.'

    # Action 2: increase temperature
    while obs['temperature'][0] < 1.:
        obs, _, _, _, _ = env.step(2)
        assert obs['temperature'][0] <= 1., 'The temperature should not increase above 1.'

    obs, _, _, _, _ = env.step(2)
    current_temperature = obs['temperature'][0]
    assert current_temperature == 1., 'The temperature should not increase above 1.'


def test_state_is_equal_to_observation():
    """Test that the state is equal to the observation."""
    env = Furnace()
    env.reset()
    done = False
    while not done:
        random_action = np.random.randint(0, env._action_space.n)
        obs, _, terminated, truncated, _ = env.step(random_action)
        done = terminated or truncated
        state = env.state
        assert np.array_equal(
            obs['PF'], state['PF'],
        ), 'The PF in the state and observation should be equal.'
        assert obs['temperature'][0] == state['temperature'][0], \
            'The temperature in the state and observation should be equal.'
        assert obs['timestep'][0] == state['timestep'][0], \
            'The timestep in the state and observation should be equal.'


def test_steps_counter():
    """Test that the steps counter is correct."""
    env = Furnace()
    env.reset()
    assert env.steps == 0, 'The steps counter should be 0.'
    done = False
    counter = 0
    while not done:
        assert env.steps == counter, 'The steps counter is not equal to the number of taken steps.'
        random_action = np.random.randint(0, env._action_space.n)
        _, _, terminated, truncated, _ = env.step(random_action)
        done = terminated or truncated
        counter += 1


def test_truncated():
    """Test that the episode is truncated."""
    env = Furnace()
    env.cfg.horizon = 10
    env = Furnace(dict(env.cfg))
    env.reset()
    done = False
    while not done:
        random_action = np.random.randint(0, env._action_space.n - 1)
        _, _, terminated, truncated, _ = env.step(random_action)
        done = terminated or truncated
    assert truncated is True, 'Truncated should be True.'
    assert env.steps == env.cfg.horizon, 'The number of steps should be equal to the horizon.'

    # Test that the episode is truncated when the stop action is not used.
    env.cfg.use_stop_action = False
    env = Furnace(dict(env.cfg))
    env.reset()
    done = False
    while not done:
        random_action = np.random.randint(0, env._action_space.n)
        _, _, terminated, truncated, _ = env.step(random_action)
        done = terminated or truncated
    assert truncated is True, 'Truncated should be True.'


def test_info():
    """
    Test that the info dictionary is correct. In particular g2 and density should be invariant
    with respect to translations of pf.
    """
    env = Furnace()
    env.reset(seed=42)
    _, _, _, _, info = env.step(0)
    assert 'g2' in info, 'The info dictionary should contain g2.'
    assert 'density' in info, 'The info dictionary should contain density.'
    assert 'energy_cost' in info, 'The info dictionary should contain energy_cost.'

    g2 = info['g2']
    density = info['density']
    energy_cost = info['energy_cost']
    # reset the environment and use np.roll to shift the PF
    del env
    env = Furnace()
    env.reset(seed=42)
    env.state['PF'] = np.roll(
        env.state['PF'], np.random.randint(1, 10000), axis=0,
    )
    env.state['PF'] = np.roll(
        env.state['PF'], np.random.randint(1, 10000), axis=1,
    )
    _, _, _, _, info = env.step(0)
    assert np.isclose(g2, info['g2']), \
        'The g2 should be invariant with respect to translations of pf.'
    assert np.isclose(density, info['density']), \
        'The density should be invariant with respect to translations of pf.'
    assert np.isclose(energy_cost, info['energy_cost']), \
        'The energy_cost should be invariant with respect to translations of pf.'


def test_state_and_dynamics():
    """
    Test that the state and dynamics is invariant to translation.

    Note: to check the translation invariance of the dynamics, we compare the end state of two
    episode:
    * the first episode starts from a random state, and it is run for n_steps,
    * the second episode starts from the same state, but the PF is shifted by a random amount, and
      it is run for n_steps. Then the final state is shifted back by the same amount.
    The obtained states should match if the dynamics is translational invariant.
    """
    env = Furnace()
    env.reset(seed=42)
    n_steps = 20

    def _generate_random_actions(n) -> list[int]:
        """Generate a random sequence of actions between 1 and 3 (inclusive)."""
        random_array = [np.random.randint(0, 3) for _ in range(n)]
        return random_array
    actions = _generate_random_actions(n_steps)
    for action in actions:
        obs, _, _, _, _ = env.step(action)

    obs_original = obs.copy()
    # reset the environment and use np.roll to shift the PF, then take n_steps steps, then
    # shift the PF back. The results should be the same as the final state of the original episode.
    del env
    env = Furnace()
    env.reset(seed=42)
    ax_0_shift = np.random.randint(1, 10000)
    ax_1_shift = np.random.randint(1, 10000)
    env.state['PF'] = np.roll(
        env.state['PF'], ax_0_shift, axis=0,
    )
    env.state['PF'] = np.roll(
        env.state['PF'], ax_1_shift, axis=1,
    )
    for action in actions:
        _, _, _, _, _ = env.step(action)

    env.state['PF'] = np.roll(
        env.state['PF'], -ax_1_shift, axis=1,
    )
    env.state['PF'] = np.roll(
        env.state['PF'], -ax_0_shift, axis=0,
    )

    assert np.allclose(env.state['PF'], obs_original['PF']), \
        'The updates of pf should be invariant with respect to translations.'


def test_reward():
    """
    Test that the reward is translational invariant.

    Note: for the test's logic see DocString of test_state_and_dynamics.
    """
    env = Furnace()
    env.reset(seed=42)
    reward_original_lst = []
    for _ in range(5):
        _, reward, _, _, _ = env.step(1)
        reward_original_lst.append(reward)
    # reset the environment and use np.roll to shift the PF
    del env
    env = Furnace()
    env.reset(seed=42)
    ax_0_shift = np.random.randint(1, 10000)
    ax_1_shift = np.random.randint(1, 10000)
    env.state['PF'] = np.roll(env.state['PF'], ax_0_shift, axis=0)
    env.state['PF'] = np.roll(env.state['PF'], ax_1_shift, axis=1)
    reward_lst = []
    for _ in range(5):
        _, reward, _, _, _ = env.step(1)
        reward_lst.append(reward)
    assert np.allclose(reward_lst, reward_original_lst), \
        'The reward should be invariant with respect to translations of pf.'
