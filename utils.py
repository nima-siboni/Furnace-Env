"""Utilities for Furnace."""
from __future__ import annotations

from pydantic import BaseConfig


class FurnaceConfig(BaseConfig):  # pylint: disable=too-few-public-methods
    """A class for furnace config."""
    horizon: int
    dimension: int
    minimum_temperature: float
    maximum_temperature: float
    desired_volume_fraction: float
    temperature_change_per_step: float
    number_of_PF_updates_per_step: int
    gamma: float
    termination_change_criterion: float
    use_termination_temperature_criterion: bool
    mobility_type: str
    g_list: list[int]
    shift_pf: float
    initial_pf_variation: float
    use_stop_action: bool
    energy_cost_per_step: float
    verbose: bool | None = False

    class Config:  # pylint: disable=too-few-public-methods
        """Forbidding extra values."""
        extra = 'forbid'
