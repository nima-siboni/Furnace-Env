"""Furnace environment compatible with gymnasium."""
from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from phase_field_physics.dynamics import Update_PF


def _load_default_config():
    return {
        'horizon': 2100,
        'dimension': 128,
        'minimum temperature': 100,
        'maximum temperature': 1000,
        'desired_volume_fraction': 0.2,
        'temperature change per step': 60,
        'number of PF updates per step': 100,
        'gamma': 1000,
        'termination_change_criterion': 0,
        'termination_temperature_criterion': 'False',
        'mobility_type': 'exp',
        'g_list': '1.0, 1.0',
        'shift_pf': -0.5,
        'initial_pf_variation': 0.01,
        'stop_action': 'True',
        'energy_cost_per_step': 0.0,
        'verbose': 'False',
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

    def __init__(self, env_config=None):
        """
        Creates a new instant of the Furnace environment

        :param : env_config which is a Dictionary with all the needed
        configurations, including:

        horizon -- the length of the experiment
        dimension -- the spatial length of the domain, i.e. the number of pixels
        minimum temperature -- min temperature of the furnace (in C)
        maximum temperature -- max temperature of the furnace (in C)
        desired_volume_fraction -- the volume fraction of the desired PF which
        is circular temperature change per step -- self explanatory
        number of PF updates per step -- self explanatory
        gamma -- the coefficient for calculation of the interface energy
        termination_change_criterion -- stop the episode if the change in PF is
         smaller that this criterion. To disable this criterion set it to 0.0.
        verbose -- self explanatory
        """
        super().__init__()
        if env_config is None:
            env_config = _load_default_config()
        # Offloading all the env configs
        self.cfg = env_config
        self.horizon = env_config['horizon']
        self.dimension = env_config['dimension']
        self.min_temperature = env_config['minimum temperature']
        self.max_temperature = env_config['maximum temperature']
        self.desired_volume_fraction = env_config['desired_volume_fraction']
        self.delta_t_not_scaled = np.float(
            env_config['temperature change per step'],
        )
        self.nr_pf_updates_per_step = env_config[
            'number of PF updates per step'
        ]
        self.gamma = env_config['gamma']
        self.termination_change_criterion = env_config[
            'termination_change_criterion'
        ]
        self.termination_temperature_criterion = env_config[
            'termination_temperature_criterion'
        ]
        self.mobility_type = env_config['mobility_type']
        self.g_list = np.array(
            [float(item) for item in env_config['g_list'].split(',')],
        )
        if 'shift_pf' in env_config:
            self.shift_pf = env_config['shift_pf']
        else:
            self.shift_pf = 0
        if 'stop_action' in env_config:
            self.stop_action = env_config['stop_action']
        else:
            self.stop_action = False
        self.initial_pf_variation = env_config['initial_pf_variation']
        self.energy_cost = env_config['energy_cost_per_step']
        self.verbose = env_config['verbose']

        # Creating scaled variables
        self.delta_t = self.delta_t_not_scaled / (
            self.max_temperature - self.min_temperature
        )

        # Sanity checks for the inputs
        if self.horizon is not None:
            assert (
                self.horizon > 0
            ), 'Can this be!? Does the game finish before the agent starts?'
        assert (
            self.min_temperature <= self.max_temperature
        ), 'The min temperature is higher than the max temperature.'

        # ------------------------------------------------------------------- #
        # 1. Lets first implement the action
        if self.stop_action:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Discrete(3)

        # ------------------------------------------------------------------- #

        # ------------------------------------------------------------------- #
        # 2. Now implementing the observation
        # In all the observation values we have the scaled values.

        self.observation_space = spaces.Dict(
            {
                'timestep': spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float,
                ),
                'temperature': spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float,
                ),
                'PF': spaces.Box(
                    low=self.shift_pf,
                    high=1 + self.shift_pf,
                    shape=(self.dimension, self.dimension, 1),
                    dtype=np.float,
                ),
            },
        )

        # The state
        self.state = None
        self.np_random = None
        # auxiliary variable for timestep
        self.steps = None
        self.steps_beyond_done = None
        self.reset()

        # Goal image
        self.not_shifted_desired_pf = self._return_desired_pf()

    def _return_desired_pf(self):
        """
        returns the (NOT SHIFTED) desired PF (a circle).
        """
        radius_2 = self.dimension * self.dimension * \
            self.desired_volume_fraction / np.pi
        not_shifted_desired_pf = np.zeros((self.dimension, self.dimension))
        x_center = self.dimension / 2.0
        y_center = self.dimension / 2.0
        for i in range(self.dimension):
            for j in range(self.dimension):
                if (i - y_center) ** 2 + (j - x_center) ** 2 < radius_2:
                    not_shifted_desired_pf[i, j] = 1.0
        return not_shifted_desired_pf

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> dict:
        """
        Resets the state.
        timestep is set to zero
        temperature is set to 0.50, better to set it to a value at which both
        phases are similarly stable.
        PF is set to random around 0.5 with tolerance of initial_pf_variation

        :return: the new state as a dictionary.
        """

        self.steps_beyond_done = False

        # random numbers between 0, 1
        tmp = (
            np.random.rand(self.dimension, self.dimension, 1)
            + 2.0 * (np.random.rand() - 0.5) * self.initial_pf_variation
        )
        tmp[tmp < 0] = 0
        tmp[tmp > 1] = 1
        pf_0 = tmp
        self.state = {
            'timestep': [0.0],
            'temperature': [0.5],
            'PF': pf_0 + self.shift_pf,
        }

        self.steps = 0
        return self.state

    def step(self, action: int) -> tuple:
        """
        One step of the Furnace environment is composed of:
        0 - increase the timestep
        1 - an initial change in the temperature
        2 - applying consequent "nr pf updates per step" (e.g. 10) steps of
        update for PF.

        :param action: the chosen action.
        :return: the common (Gym) output of the step function with s, r, done,
         info.
        In particular our info includes g_2, density, and energy cost for easier
         further analysis.
        """

        # make a deep copy of the state
        obs = copy.deepcopy(self.state)

        # 0 -- increase the time-step
        self.steps += 1
        obs['timestep'] = [np.float(self.steps) / self.horizon]
        if self.steps == self.horizon:
            done = True
            reward, energy_cost = self._calculate_reward(new_state=obs)
            self.state = obs
            return obs, reward, done, {}
        # --------------------------------------

        # 1 -- implement the actions
        # If we get up to here it means that the steps are in the range

        # set the done
        done = False

        # stop the process, freeze!
        if action == 3:
            done = True
            reward, energy_cost = self._calculate_reward(new_state=obs)
            self.state = obs
            return obs, reward, done, {}

        if action == 0:
            # decrease the temperature
            obs['temperature'][0] -= self.delta_t
            # Do not terminate if the temperature goes below the minimum
            if obs['temperature'][0] < \
                    self.observation_space['temperature'].low[0]:
                obs['temperature'] = self.observation_space['temperature'].low
                if self.termination_temperature_criterion:
                    done = True
                    reward, energy_cost = self._calculate_reward(new_state=obs)
                    self.state = obs
                    return obs, reward, done, {}

        if action == 2:
            # increase the temperature
            obs['temperature'][0] += self.delta_t
            # Do not terminate if the temperature goes above the maximum
            if obs['temperature'][0] >\
                    self.observation_space['temperature'].high[0]:
                obs['temperature'] = self.observation_space['temperature'].high
                if self.termination_temperature_criterion:
                    done = True
                    reward, energy_cost = self._calculate_reward(new_state=obs)
                    self.state = obs
                    return obs, reward, done, {}

        if action == 1:
            # Do not change the temperature
            obs['temperature'][0] += 0.0

        # Update phi and calculate the rewards
        phi = obs['PF'][:, :, 0] - self.shift_pf
        temperature = self.min_temperature + obs['temperature'][0] * (
            self.max_temperature - self.min_temperature
        )
        phi, dphi, g_2 = Update_PF(
            phi,
            temperature,
            self.nr_pf_updates_per_step,
            self.g_list,
            self.mobility_type,
        )
        obs['PF'] = np.expand_dims(phi, axis=-1) + self.shift_pf

        if self.termination_change_criterion is not None:
            done = np.max(np.abs(dphi)) < self.termination_change_criterion

        reward, energy_cost = self._calculate_reward(new_state=obs)
        self.state = obs
        return (
            obs,
            reward,
            done,
            {'g2': g_2, 'density': np.mean(phi), 'energy_cost': energy_cost},
        )

    def _calculate_reward(self, new_state: dict) -> tuple[float, float]:
        """
        This function calculates the immediate reward which should depend on
        the changes of the structure.

        :note: The previous state is accessible via self.state
        :param new_state: The current state
        :return: the immediate reward, and energy cost (the energy cost is not
        used for training, but for analysis).
        """
        # new overlap
        phi = new_state['PF'][:, :, 0] - self.shift_pf
        centered_phi_lst = []
        shifted_phi_lst = self._translate_half_box(phi)
        for shifted_phi in shifted_phi_lst:
            translated_to_center_lst = self._translate_to_the_center(
                shifted_phi,
            )
            for phi in translated_to_center_lst:
                centered_phi_lst.append(phi)
        iou_lst = [self._return_iou(phi) for phi in centered_phi_lst]
        new_max_iou = np.max(iou_lst)

        # old overlap
        phi = self.state['PF'][:, :, 0] - self.shift_pf
        centered_phi_lst = []
        shifted_phi_lst = self._translate_half_box(phi)
        for shifted_phi in shifted_phi_lst:
            translated_to_center_lst = self._translate_to_the_center(
                shifted_phi,
            )
            for phi in translated_to_center_lst:
                centered_phi_lst.append(phi)
        iou_lst = [self._return_iou(phi) for phi in centered_phi_lst]
        old_max_iou = np.max(iou_lst)

        reward = new_max_iou - old_max_iou

        # energy cost
        degrees_above_room_t = (self.min_temperature - 22) / (
            self.max_temperature - self.min_temperature
        )
        energy_cost = (
            new_state['temperature'][0] + degrees_above_room_t
        ) * self.energy_cost
        reward -= energy_cost

        return reward, energy_cost

    def _translate_half_box(self, phase_field):
        """
        Translate the PF half box in direction of x and y, and x-and-y.

        :param phase_field: the phase field
        :return: a list consisting of the original pf and the shifted ones.
        """
        current_pf_0 = phase_field
        current_pf_1 = np.roll(current_pf_0, int(self.dimension / 2.0), axis=0)
        current_pf_2 = np.roll(current_pf_0, int(self.dimension / 2.0), axis=1)
        current_pf_3 = np.roll(current_pf_2, int(self.dimension / 2.0), axis=0)
        return [current_pf_0, current_pf_1, current_pf_2, current_pf_3]

    def _translate_to_the_center(self, phase_field: np.ndarray) -> list:
        """Shifts the image such that the center of mass of the phase 1 is at
        the middle of the image.

        This shift is applied to (half box) translated versions of the original
         PF.

        :param phase_field: the original PF
        :return: a list of PFs with their center of mass shifted to
        """
        epsilon = 1e-5
        x_cm = 0
        y_cm = 0
        for i in range(self.dimension):
            for j in range(self.dimension):
                x_cm += j * phase_field[i, j]
                y_cm += i * phase_field[i, j]
        if np.sum(phase_field) > epsilon:
            x_cm = x_cm / np.sum(phase_field)
            y_cm = y_cm / np.sum(phase_field)
        else:
            x_cm = self.dimension / 2.0
            y_cm = self.dimension / 2.0

        shift_x = int(self.dimension / 2.0 - x_cm)
        shift_y = int(self.dimension / 2.0 - y_cm)
        current_pf_0 = np.roll(phase_field, shift=shift_x, axis=1)
        current_pf_0 = np.roll(current_pf_0, shift=shift_y, axis=0)

        current_pf_1 = np.roll(current_pf_0, int(self.dimension / 2.0), axis=0)
        current_pf_2 = np.roll(current_pf_0, int(self.dimension / 2.0), axis=1)
        current_pf_3 = np.roll(current_pf_2, int(self.dimension / 2.0), axis=0)

        return [current_pf_0, current_pf_1, current_pf_2, current_pf_3]

    def _return_iou(self, image: np.ndarray) -> float:
        """
        Returns the (modified) overlap of image and the self.desired_pf.
        """
        desired_pf = self.not_shifted_desired_pf
        corrected_overlap = 3 * desired_pf * image - desired_pf - image
        corrected_overlap = np.sum(corrected_overlap)

        vmax_overlap = np.sum(desired_pf)  # true estimate
        vmin_overlap = -1.0 * vmax_overlap  # estimate

        corrected_overlap = (
            vmin_overlap
            if corrected_overlap < vmin_overlap else corrected_overlap
        )
        corrected_overlap = corrected_overlap / \
            vmax_overlap  # in range [-1, 1]

        corrected_overlap += 1  # in range [0, 2]

        return corrected_overlap

    def _set_pf(self, phase_field: np.ndarray) -> dict:
        """
        Sets the desired PF.

        :return: the new state
        """
        assert np.max(phase_field) <= 1.0
        assert np.min(phase_field) >= 0.0
        self.state['PF'][:, :, 0] = phase_field + self.shift_pf
        return self.state

    def render(self):
        raise NotImplementedError
