"""Utilities for Furnace."""
from __future__ import annotations

from pydantic import BaseModel  # pylint: disable=no-name-in-module


class FurnaceConfig(BaseModel):  # pylint: disable=too-few-public-methods
    """A class for furnace config."""
    horizon: int
    dimension: int
    minimum_temperature: float
    maximum_temperature: float
    desired_volume_fraction: float
    temperature_change_per_step: float
    number_of_pf_updates_per_step: int
    gamma: float
    termination_change_criterion: float | None
    use_termination_temperature_criterion: bool
    mobility_type: str
    g_list: list[int]
    shift_pf: float | None = 0.0
    initial_pf_variation: float
    use_stop_action: bool
    energy_cost_per_step: float
    verbose: bool | None = False

    class Config:  # pylint: disable=too-few-public-methods
        """Forbidding extra values."""
        extra = 'forbid'


def config_checks(config: FurnaceConfig):
    """Check config."""
    assert config.horizon > 0, 'Can this be!? Does the game finish before the agent starts?'
    assert config.maximum_temperature > config.minimum_temperature, \
        'max temperature should be be higher than min temperature.'
