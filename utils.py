"""Utilities for Furnace."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel  # pylint: disable=no-name-in-module


class FurnaceConfig(BaseModel):  # pylint: disable=too-few-public-methods
    """A class for furnace config."""
    horizon: int  # time horizon of the environment
    dimension: int  # dimensions of the environment's box
    minimum_temperature: float  # minimum temperature of the environment in C
    maximum_temperature: float  # maximum temperature of the environment in C
    desired_volume_fraction: float  # desired volume fraction
    temperature_change_per_step: float  # temperature change per step
    number_of_pf_updates_per_step: int  # number of phase field updates per step
    gamma: float  # interface energy constant
    # termination pf change criterion, if the change
    termination_change_criterion: Optional[float]
    # is less than this value, the episode terminates
    # if true, the episode terminates if the temperature
    use_termination_temperature_criterion: bool
    # goes out of the range [minimum_temperature, maximum_temperature]
    mobility_type: str  # mobility type
    g_list: list[int]  # list of g values
    shift_pf: Optional[float] = 0.0  # a constant shift of pf values
    initial_pf_variation: float  # initial pf variation
    use_stop_action: bool  # if true, the agent can stop the furnace
    # energy cost per step per degree above the room temperature
    energy_cost_per_step: float
    room_temperature: Optional[float] = 22.0  # room temperature in C
    # if true, the environment prints some information
    verbose: Optional[bool] = False

    class Config:  # pylint: disable=too-few-public-methods
        """Forbidding extra values."""
        extra = 'forbid'


def config_checks(config: FurnaceConfig):
    """Check config."""
    assert config.horizon > 0, 'Can this be!? Does the game finish before the agent starts?'
    assert config.maximum_temperature > config.minimum_temperature, \
        'max temperature should be be higher than min temperature.'
