"""Furnace environment compatible with gymnasium."""
from __future__ import annotations

import copy
from typing import Any
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from furnace_utils import config_checks
from furnace_utils import FurnaceConfig
from phase_field_physics.dynamics import Update_PF


def _load_default_config():
    return {
        'horizon': 2100,
        'dimension': 120,
        'minimum_temperature': 100,
        'maximum_temperature': 1000,
        'desired_volume_fraction': 0.2,
        'temperature_change_per_step': 60,
        'number_of_pf_updates_per_step': 100,
        'gamma': 1000,
        'termination_change_criterion': 0.,
        'use_termination_temperature_criterion': False,
        'mobility_type': 'exp',
        'g_list': [1.0, 1.0],
        'shift_pf': -0.5,
        'initial_pf_variation': 0.01,
        'use_stop_action': True,
        'energy_cost_per_step': 0.0,
        'verbose': False,
    }


class Furnace(gym.Env):  # pylint: disable=too-many-instance-attributes
    """
    Furnace environment:
    The environment designed to experiment with heat treatment
    Actions:
    Type: Discrete (4)

    Description                                     Shape              Range
    ------------                                    -------            -------
    The change in temperature or terminate          (1,)               [0, 1]

    Num   Action
    0     Reduce the temperature
    1     Do not change the temperature
    2     Increase the temperature
    3     Stop the process

    Observation:
    Type: Dict

    Observation     Key               Shape              Range
    -----------     -----             -------            -------
    timestep        'timestep'        (1,)               [0, 1]
    Temperature     'temperature'     (1,)               [0, 1]
    Phase Field     'PF'              (horizon, horizon, 1)          each element in [0, 1]

    Reward:
    All the (s, a) have reward zero unless we reach the terminal state.
     In that case, the reward has two parts:

    Starting State:
    A random state in of pf and (scaled) temperature 0.5

    Episode Termination:
    Either:
       * Reach the number of steps reaches horizon,
       * The change in dphi is smaller than a value (if the give value is 0.0
       this condition is effectively ignored).
       * The temperature is out of range; this condition is active only if
        termination_temperature_criterion = True.
       In the case where the parameter is False the temperature is set to the
        corresponding boundary value if it gets out of bounds.
       * If the action 3 is chosen.
    """

    def __init__(self, env_config: Optional[dict[str, Any]] = None) -> None:
        """
        Creates a new instant of the Furnace environment

        Args:
            env_config: a dictionary with the configuration of the environment. See FurnaceConfig
            for more details.
        """
        super().__init__()
        if env_config is None:
            env_config = _load_default_config()
        self.cfg = FurnaceConfig(**env_config)

        # Sanity checks for the inputs
        config_checks(self.cfg)

        self._action_space = self._create_action_space(self.cfg)

        self._observation_space = self._create_observation_space(self.cfg)

        # Auxiliary variables: scaled delta t and not shifted desired pf
        self._delta_t = self.cfg.temperature_change_per_step / (
            self.cfg.maximum_temperature - self.cfg.minimum_temperature
        )
        self._not_shifted_desired_pf = self._return_desired_pf()

        # More auxiliary variables
        self._state = None
        self._np_random = None
        self._steps = None
        self._steps_beyond_done = None
        self._old_correlation = 0.0
        self._desired_pf_fourier_transform = None

    @staticmethod
    def _create_observation_space(config: FurnaceConfig) -> spaces.Dict:
        """Return the observation space.

        Notes: In all the observation values we have the scaled values.

        Args:
            config: FurnaceConfig.
        """
        observation_space = spaces.Dict(
            {
                'timestep': spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=float,
                ),
                'temperature': spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=float,
                ),
                'PF': spaces.Box(
                    low=config.shift_pf,
                    high=1 + config.shift_pf,
                    shape=(config.dimension, config.dimension, 1),
                    dtype=float,
                ),
            },
        )
        return observation_space

    def _return_desired_pf(self) -> np.ndarray:
        """
        Return the desired PF.

        Note: the PF is not shifted
        Note: the PF is in shape of a circle.

        Returns: the desired PF.
        """
        radius_2 = self.cfg.dimension * self.cfg.dimension * \
            self.cfg.desired_volume_fraction / np.pi
        not_shifted_desired_pf = np.zeros(
            (self.cfg.dimension, self.cfg.dimension),
        )
        x_center = self.cfg.dimension / 2.0
        y_center = self.cfg.dimension / 2.0
        for i in range(self.cfg.dimension):
            for j in range(self.cfg.dimension):
                if (i - y_center) ** 2 + (j - x_center) ** 2 < radius_2:
                    not_shifted_desired_pf[i, j] = 1.0
        return not_shifted_desired_pf

    def _return_desired_pf_fourier_transform(self) -> np.ndarray:
        """
        Return the normalized FFT of the desired PF.

        Note: the FFT is normalized by subtracting the mean and dividing by the standard deviation.
        This is helpful for calculation of the correlation between the FFT of the desired PF and the
        FFT of the current PF.

        Returns: the normalized FFT of the desired PF.
        """
        if self._desired_pf_fourier_transform is None:
            fourier_transform = np.abs(
                np.fft.fft2(self._not_shifted_desired_pf),
            )
            # find the standard deviation and mean of the Fourier transform
            std = np.std(fourier_transform)
            mean = np.mean(fourier_transform)
            # normalize the Fourier transform
            fourier_transform_normalized = (fourier_transform - mean) / std
            self._desired_pf_fourier_transform = fourier_transform_normalized
        return self._desired_pf_fourier_transform

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> \
            tuple[spaces.Dict, dict]:
        """
        Resets the environment.

        Notes: The followings are done:
        timestep is set to zero.
        temperature is set to 0.50, better to set it to a value at which both phases are similarly
        stable.
        PF is set to random around 0.5 with tolerance of initial_pf_variation.
        Sets the old_correlation to zero.

        Args:
            seed: the seed for the random number generator.
            options: a dictionary with additional options.

        Returns:
             the new state as a dictionary.
             the new info as a dictionary which has None for all the values.
        """

        self._steps_beyond_done = False

        if seed is not None:
            np.random.seed(seed)
        # random phase field values between 0, 1
        tmp = np.random.rand(self.cfg.dimension, self.cfg.dimension, 1) + \
            2.0 * (np.random.rand() - 0.5) * self.cfg.initial_pf_variation

        tmp[tmp < 0] = 0
        tmp[tmp > 1] = 1
        pf_0 = tmp

        self.state = {
            'timestep': [0.0],
            'temperature': [0.5],
            'PF': pf_0 + self.cfg.shift_pf,
        }

        self.steps = 0
        self._old_correlation = 0.0
        self._desired_pf_fourier_transform = None
        return self.state, {'g2': None, 'density': None, 'energy_cost': None}

    def step(self, action: int) -> tuple:
        """
        Step the environment.

        Notes: One step of the Furnace environment is composed of:
        0 - increase the timestep,
        1 - a change in the temperature,
        2 - applying consequent "nr pf updates per step" (e.g. 10) steps of update for PF.

        Args:
            action: the chosen action.

        Returns:
            observation: the new phase field, temperature, and timestep.
            reward: the reward for the action.
            terminated: True if the episode is terminated.
            truncated: if the horizon is reached.
            info: a dictionary with additional information about the environment. In particular,
            it includes g_2, density, and energy cost for easier further analysis.
        """

        obs = copy.deepcopy(self.state)

        # 0 -- increase the timestep
        self.steps += 1
        obs['timestep'] = [float(self.steps) / self.cfg.horizon]

        # 0.5 -- check if the horizon is reached
        if self.steps == self.cfg.horizon:
            truncated = True
            terminated = False
            reward, energy_cost = self._calculate_reward(new_state=obs)
            self.state = obs
            return obs, reward, terminated, truncated, {}
        # --------------------------------------

        truncated = False
        terminated = False
        # 1 -- implement the actions
        # If we get up to here it means that the steps are in the range

        # stop the process, freeze!
        if action == 3:
            terminated = True
            reward, energy_cost = self._calculate_reward(new_state=obs)
            self.state = obs
            return obs, reward, terminated, truncated, {}

        obs['temperature'][0], out_of_bound_temperature = self._update_temperature(
            action=action,
        )

        if out_of_bound_temperature and self.cfg.use_termination_temperature_criterion:
            terminated = True
            reward, energy_cost = self._calculate_reward(new_state=obs)
            self.state = obs
            return obs, reward, terminated, truncated, {}

        # Update phi and calculate the rewards
        obs, density, max_abs_dphi, g_2 = self._update_pf(obs)

        if self.cfg.termination_change_criterion is not None:
            terminated = max_abs_dphi < self.cfg.termination_change_criterion

        reward, energy_cost = self._calculate_reward(new_state=obs)

        info = {'g2': g_2, 'density': density, 'energy_cost': energy_cost}
        self.state = obs
        return obs, reward, terminated, truncated, info

    def _update_pf(self, obs: spaces.Dict) -> tuple[dict, float, float, float]:
        """
        Update the phase field.

        Args:
            obs: the current state.

        Returns:
            new_pf: the new phase field.
            g_2: the g_2 value.
        """

        phi = obs['PF'][:, :, 0] - self.cfg.shift_pf
        temperature = self.cfg.minimum_temperature + obs['temperature'][0] * (
            self.cfg.maximum_temperature - self.cfg.minimum_temperature
        )

        phi, dphi, g_2 = Update_PF(
            phi,
            temperature,
            self.cfg.number_of_pf_updates_per_step,
            self.cfg.g_list,
            self.cfg.mobility_type,
        )
        obs['PF'] = np.expand_dims(phi, axis=-1) + self.cfg.shift_pf
        # TODO: check the types returned from UPDATE_PF # pylint: disable=fixme
        return obs, float(np.mean(phi)), float(np.max(np.abs(dphi))), float(g_2)

    def _update_temperature(self, action: int) -> tuple[float, bool]:
        """
        Return the updated temperature and whether the temperature is out of bounds.

        Args:
            action: the chosen action.

        Returns:
            temperature: the new temperature.
            out_of_bounds: True if the temperature is out of bounds.
        """
        temperature = self.state['temperature'][0]
        out_of_bounds = False
        if action == 0:
            temperature -= self._delta_t
        if action == 2:
            temperature += self._delta_t
        if action == 1:
            temperature += 0.0

        # TODO: refactor this using np.clip # pylint: disable=fixme
        if temperature < self._observation_space['temperature'].low[0]:
            temperature = self._observation_space['temperature'].low[0]
            out_of_bounds = True
        if temperature > self._observation_space['temperature'].high[0]:
            temperature = self._observation_space['temperature'].high[0]
            out_of_bounds = True

        return temperature, out_of_bounds

    def _calculate_reward(self, new_state: spaces.Dict) -> tuple[float, float]:
        """
        Return the reward and energy cost.

        Notes:
            The previous state is accessible via self.state.
            The reward is the sum of two values:
             the correlation between the normalized Fourier transform of the desired PF and the
             normalized Fourier transform of the current PF, and
             the negative of energy cost in this step.

        Args:
            new_state: the new state.

        Returns:
            reward: the reward.
            energy_cost: the energy cost; it is not used for training, but for analysis.
        """
        # Take the Fourier transform
        fourier_transform = np.abs(np.fft.fft2(new_state['PF'][:, :, 0]))
        # find the standard deviation and mean of the Fourier transform
        std = np.std(fourier_transform)
        mean = np.mean(fourier_transform)
        normalized_fourier_transform = (fourier_transform - mean) / std
        # calculate the correlation between the normalized Fourier transform of the desired PF and
        # the normalized Fourier transform of the current PF
        correlation = float(
            np.mean(
                normalized_fourier_transform * self._return_desired_pf_fourier_transform(),
            ),
        )
        # TODO: fix this, is negative correlation worse than 0 correlation? # pylint: disable=fixme
        # value of self._old_correlation
        correlation = 0 if correlation < 0 else correlation
        assert -1.0 <= correlation <= 1.0, f"correlation between the desired PF's FFT magnitude " \
                                           f"and the current PF's FFT magnitude is {correlation}," \
                                           f' it should be between -1 and 1.'
        reward = correlation - self._old_correlation
        self._old_correlation = correlation
        # energy cost
        energy_cost = self._calculate_energy_cost(new_state=new_state)
        reward -= energy_cost

        return reward, energy_cost

    def _calculate_energy_cost(self, new_state: spaces.Dict) -> float:
        """
        Return the energy cost.

        Args:
            new_state: the new state.

        Returns:
            energy_cost: the energy cost.
        """

        unscaled_temperature = self.cfg.minimum_temperature + new_state['temperature'][0] * (
            self.cfg.maximum_temperature - self.cfg.minimum_temperature
        )
        energy_cost = self.cfg.energy_cost_per_step * \
            (unscaled_temperature - self.cfg.room_temperature)

        return energy_cost

    def _set_pf(self, phase_field: np.ndarray) -> spaces.Dict:
        """
        Set the desired PF after shifting it and return the new state.

        Args:
            phase_field: the new phase field.
        Returns:
            the updated state.
        """
        assert np.max(phase_field) <= 1.0
        assert np.min(phase_field) >= 0.0
        self.state['PF'][:, :, 0] = phase_field + self.cfg.shift_pf
        return self.state

    def render(self):
        raise NotImplementedError

    @staticmethod
    def _create_action_space(config: FurnaceConfig) -> spaces.Discrete:
        """
        Return the action space.

        Args:
            config: FurnaceConfig from which we only need use_stop_action.

        Returns:
            a discrete space with size 3 (if use_stop_action) or 4 (if !use_stop_action).
        """
        return spaces.Discrete(4) if config.use_stop_action else spaces.Discrete(3)

    @property
    def steps(self) -> int:
        """Return steps."""
        return self._steps

    @property
    def state(self) -> spaces.Dict:
        """Return state."""
        return self._state

    @state.setter
    def state(self, value: spaces.Dict):
        """Set state."""
        self._state = value

    @steps.setter
    def steps(self, value: int):
        """Set steps."""
        self._steps = value

    @property
    def observation_space(self) -> spaces.Dict:
        """Return the observation space."""
        return self._observation_space

    @property
    def action_space(self) -> spaces.Discrete:
        """Return the action space."""
        return self._action_space
