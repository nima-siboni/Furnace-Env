from __future__ import annotations

import copy

import gym
import numpy as np
from gym import spaces
TerminationAction: int = 3
from phase_field_physics.dynamics import Update_PF


def _load_default_config():
    return {
        'N': 2100,
        'L': 128,
        'minimum temperature': 100,
        'maximum temperature': 1000,
        'desired_volume_fraction': 0.2,
        'temperature change per step': 60,
        'number of PF updates per step': 100,
        'gamma': 1000,
        'termination_change_criterion': 0,
        'termination_temperature_criterion': 'False',
        'mobility_type': 'exp',
        'G_list': '1.0, 1.0',
        'shift_PF': -0.5,
        'initial_PF_variation': 0.01,
        'stop_action': 'True',
        'energy_cost_per_step': 0.0,
        'verbose': 'False',
    }


class Furnace(gym.Env):
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
    Phase Field     'PF'              (N, N, 1)          each element in [0, 1]

    Reward:
    All the (s, a) have reward zero unless we reach the terminal state.
     In that case, the reward has two parts:

    Starting State:
    A random state in of pf and (scaled) temperature 0.5

    Episode Termination:
    Either:
       * Reach the number of steps reaches N,
       * The change in dphi is smaller than a value (if the give value is 0.0
       this condition is effectively ignored).
       * The temperature is out of range; this condition is active only if
        termination_temperature_criterion = True.
       In the case where the parameter is False the temperature is set to the
        corresponding boundary value if it gets out of bounds.
       * If the action 3 is chosen.
    """

    def __init__(self, env_config=None):

        super().__init__()
        """
        Creates a new instant of the Furnace environment

        :param : env_config which is a Dictionary with all the needed
        configurations, including:

        N -- the length of the experiment
        L -- the spatial length of the domain, i.e. the number of pixels
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
        if env_config is None:
            env_config = _load_default_config()
        # Offloading all the env configs
        self.cfg = env_config
        self.N = env_config['N']
        self.L = env_config['L']
        self.min_temperature = env_config['minimum temperature']
        self.max_temperature = env_config['maximum temperature']
        self.desired_volume_fraction = env_config['desired_volume_fraction']
        self.delta_T_not_scaled = np.float(
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
        self.G_list = np.array(
            [float(item) for item in env_config['G_list'].split(',')],
        )
        if 'shift_PF' in env_config:
            self.shift_PF = env_config['shift_PF']
        else:
            self.shift_PF = 0
        if 'stop_action' in env_config:
            self.stop_action = env_config['stop_action']
        else:
            self.stop_action = False
        self.initial_PF_variation = env_config['initial_PF_variation']
        self.energy_cost = env_config['energy_cost_per_step']
        self.verbose = env_config['verbose']

        # Creating scaled variables
        self.delta_T = self.delta_T_not_scaled / (
            self.max_temperature - self.min_temperature
        )

        # Sanity checks for the inputs
        if self.N is not None:
            assert (
                self.N > 0
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
                    low=self.shift_PF,
                    high=1 + self.shift_PF,
                    shape=(self.L, self.L, 1),
                    dtype=np.float,
                ),
            },
        )

        # The state
        self.state = None
        self.np_random = None
        # auxiliary variable for timestep
        self.steps = None
        self.seed()
        self.steps_beyond_done = None
        self.reset()

        # Goal image
        self.not_shifted_desired_PF = self._return_desired_PF()

    def _return_desired_PF(self):
        """
        returns the (NOT SHIFTED) desired PF (a circle).
        """
        radius_2 = self.L * self.L * self.desired_volume_fraction / np.pi
        not_shifted_desired_PF = np.zeros((self.L, self.L))
        x_center = self.L / 2.0
        y_center = self.L / 2.0
        for i in range(self.L):
            for j in range(self.L):
                if (i - y_center) ** 2 + (j - x_center) ** 2 < radius_2:
                    not_shifted_desired_PF[i, j] = 1.0
        return not_shifted_desired_PF

    def seed(self, seed=None) -> list:
        """
        Set the seed of the numpy random generator and returns the seed.

        :param seed: the seed # practically never used
        :return: the seed as a list.
        """
        np.random.seed(seed)
        return [seed]

    def reset(self) -> dict:
        """
        Resets the state.
        timestep is set to zero
        temperature is set to 0.50, better to set it to a value at which both
        phases are similarly stable.
        PF is set to random around 0.5 with tolerance of initial_PF_variation

        :return: the new state as a dictionary.
        """

        self.steps_beyond_done = False

        # random numbers between 0, 1
        tmp = (
            np.random.rand(self.L, self.L, 1)
            + 2.0 * (np.random.rand() - 0.5) * self.initial_PF_variation
        )
        tmp[tmp < 0] = 0
        tmp[tmp > 1] = 1
        PF0 = tmp
        self.state = {
            'timestep': [0.0],
            'temperature': [0.5],
            'PF': PF0 + self.shift_PF,
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
        In particular our info includes g2, density, and energy cost for easier
         further analysis.
        """

        # make a deep copy of the state
        obs = copy.deepcopy(self.state)

        assert TerminationAction == 3, 'The termination action is not 3.'
        # 0 -- increase the time-step
        self.steps += 1
        obs['timestep'] = [np.float(self.steps) / self.N]
        if self.steps == self.N:
            done = True
            # pass action the termination action, to get the final reward.
            reward, energy_cost = self._calculate_reward(new_state=obs, action=TerminationAction)
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
            reward, energy_cost = self._calculate_reward(new_state=obs, action=action)
            self.state = obs
            return obs, reward, done, {"reward_for_termination": reward}

        if action == 0:
            # decrease the temperature
            obs['temperature'][0] -= self.delta_T
            # Do not terminate if the temperature goes below the minimum
            if obs['temperature'][0] < \
                    self.observation_space['temperature'].low[0]:
                obs['temperature'] = self.observation_space['temperature'].low
                if self.termination_temperature_criterion:
                    done = True
                    reward, energy_cost = self._calculate_reward(new_state=obs, action=action)
                    # calculate the reward for the termination action and add it to the info
                    terminated_reward, _ = self._calculate_reward(new_state=obs, action=TerminationAction)
                    self.state = obs
                    return obs, reward, done, {"reward_for_termination": terminated_reward}

        if action == 2:
            # increase the temperature
            obs['temperature'][0] += self.delta_T
            # Do not terminate if the temperature goes above the maximum
            if obs['temperature'][0] >\
                    self.observation_space['temperature'].high[0]:
                obs['temperature'] = self.observation_space['temperature'].high
                if self.termination_temperature_criterion:
                    done = True
                    reward, energy_cost = self._calculate_reward(new_state=obs, action=action)
                    # calculate the reward for the termination action and add it to the info
                    terminated_reward, _ = self._calculate_reward(new_state=obs, action=TerminationAction)
                    self.state = obs
                    return obs, reward, done, {"reward_for_termination": terminated_reward}

        if action == 1:
            # Do not change the temperature
            obs['temperature'][0] += 0.0

        # Update phi and calculate the rewards
        phi = obs['PF'][:, :, 0] - self.shift_PF
        temperature = self.min_temperature + obs['temperature'][0] * (
            self.max_temperature - self.min_temperature
        )
        phi, dphi, g2 = Update_PF(
            phi,
            temperature,
            self.nr_pf_updates_per_step,
            self.G_list,
            self.mobility_type,
        )
        obs['PF'] = np.expand_dims(phi, axis=-1) + self.shift_PF

        if self.termination_change_criterion is not None:
            if np.max(np.abs(dphi)) < self.termination_change_criterion:
                done = True
            else:
                done = False

        reward, energy_cost = self._calculate_reward(new_state=obs, action=action)
        # calculate the reward for the termination action and add it to the info
        terminated_reward, _ = self._calculate_reward(new_state=obs, action=TerminationAction)
        self.state = obs
        info = {'g2': g2,
                 'density': np.mean(phi),
                 'energy_cost': energy_cost,
                 "reward_for_termination": terminated_reward}
        return obs, reward, done, info

    def _calculate_reward(self, new_state: dict, action: int) -> tuple[float, float]:
        """
        This function calculates the immediate reward:
        - if the action is 3 (termination) the reward is the reward of how close is the final state to the target + the
        energy cost of the immediate step.
        - if the action is not the terminated action the reward is the immediate energy cost of the action

        :note: The previous state is accessible via self.state

        :param new_state: The current state
        :param action: The action taken
        :return: the immediate reward, and energy cost (the energy cost is not
        used for training, but for analysis).
        """

        # Let's first calculate the energy cost which is the same for all actions
        degrees_above_room_T = (self.min_temperature - 22) / (
            self.max_temperature - self.min_temperature
        )
        energy_cost = (
            new_state['temperature'][0] + degrees_above_room_T
        ) * self.energy_cost
        energy_reward = -energy_cost

        if action != 3:
            reward = energy_reward
            return reward, energy_cost

        # if the action is to terminate the process, then the reward is due to
        # the energy cost and the micro-structure

        phi = new_state['PF'][:, :, 0] - self.shift_PF
        centered_phi_lst = []
        shifted_phi_lst = self._translate_half_box(phi)
        for shifted_phi in shifted_phi_lst:
            translated_to_center_lst = self._translate_to_the_center(
                shifted_phi,
            )
            for phi in translated_to_center_lst:
                centered_phi_lst.append(phi)
        IoU_lst = [self._IoU(phi) for phi in centered_phi_lst]
        new_max_IoU = np.max(IoU_lst)

        microstructure_reward = new_max_IoU

        # total reward: add the energy reward to the microstructure reward
        reward = microstructure_reward + energy_reward

        return reward, energy_cost

    def _translate_half_box(self, PF):
        """
        Translate the PF half box in direction of x and y, and x-and-y.

        :param PF: the phase field
        :return: a list consisting of the original pf and the shifted ones.
        """
        current_PF_0 = PF
        current_PF_1 = np.roll(current_PF_0, int(self.L / 2.0), axis=0)
        current_PF_2 = np.roll(current_PF_0, int(self.L / 2.0), axis=1)
        current_PF_3 = np.roll(current_PF_2, int(self.L / 2.0), axis=0)
        return [current_PF_0, current_PF_1, current_PF_2, current_PF_3]

    def _translate_to_the_center(self, PF: np.ndarray) -> list:
        """Shifts the image such that the center of mass of the phase 1 is at
        the middle of the image.

        This shift is applied to (half box) translated versions of the original
         PF.

        :param PF: the original PF
        :return: a list of PFs with their center of mass shifted to
        """
        epsilon = 1e-5
        x_cm = 0
        y_cm = 0
        for i in range(self.L):
            for j in range(self.L):
                x_cm += j * PF[i, j]
                y_cm += i * PF[i, j]
        if np.sum(PF) > epsilon:
            x_cm = x_cm / np.sum(PF)
            y_cm = y_cm / np.sum(PF)
        else:
            x_cm = self.L / 2.0
            y_cm = self.L / 2.0

        shift_x = int(self.L / 2.0 - x_cm)
        shift_y = int(self.L / 2.0 - y_cm)
        current_PF_0 = np.roll(PF, shift=shift_x, axis=1)
        current_PF_0 = np.roll(current_PF_0, shift=shift_y, axis=0)

        current_PF_1 = np.roll(current_PF_0, int(self.L / 2.0), axis=0)
        current_PF_2 = np.roll(current_PF_0, int(self.L / 2.0), axis=1)
        current_PF_3 = np.roll(current_PF_2, int(self.L / 2.0), axis=0)

        return [current_PF_0, current_PF_1, current_PF_2, current_PF_3]

    def _IoU(self, image: np.ndarray) -> float:
        """
        Returns the (modified) overlap of image and the self.desired_PF.
        """
        desired_PF = self.not_shifted_desired_PF
        corrected_overlap = 3 * desired_PF * image - desired_PF - image
        corrected_overlap = np.sum(corrected_overlap)

        vmax_overlap = np.sum(desired_PF)  # true estimate
        vmin_overlap = -1.0 * vmax_overlap  # estimate

        corrected_overlap = (
            vmin_overlap
            if corrected_overlap < vmin_overlap else corrected_overlap
        )
        corrected_overlap = corrected_overlap / \
            vmax_overlap  # in range [-1, 1]

        corrected_overlap += 1  # in range [0, 2]

        return corrected_overlap

    def _set_pf(self, pf: np.ndarray) -> dict:
        """
        Sets the desired PF.

        :return: the new state
        """
        assert np.max(pf) <= 1.0
        assert np.min(pf) >= 0.0
        self.state['PF'][:, :, 0] = pf + self.shift_PF
        return self.state
