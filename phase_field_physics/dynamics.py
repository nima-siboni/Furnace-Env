# Allen-Cahn phase-field model
from __future__ import annotations

import numpy as np


def Calc_Del2(Phi: np.ndarray) -> np.ndarray:
    """
    Calculates the Laplacian of a 2D data with finite difference (periodic BC)

    :param Phi: the input phase field
    :return: the Laplacian
    """
    dx = 1
    dy = 1
    PhiFX = np.roll(Phi, 1, axis=0)
    PhiBX = np.roll(Phi, -1, axis=0)
    PhiFY = np.roll(Phi, 1, axis=1)
    PhiBY = np.roll(Phi, -1, axis=1)
    return (PhiFX + PhiFY + PhiBX + PhiBY - 4 * Phi) / (dx * dy)


def Calc_Interface_Energy(Phi: np.ndarray, gamma: float) -> float:
    """
    Magnitude of the Gradient of a 2D data with finite difference (periodic BC)

    :param Phi: the input phase field
    :param gamma: the interface energy constant
    :return: the interface energy
    """
    dx = 1
    dy = 1
    PhiFX = np.roll(Phi, 1, axis=0)
    PhiBX = np.roll(Phi, -1, axis=0)
    PhiFY = np.roll(Phi, 1, axis=1)
    PhiBY = np.roll(Phi, -1, axis=1)
    gradient_2 = np.power((PhiFX + PhiBX - 2 * Phi) / (2 * dx), 2) + np.power(
        (PhiFY + PhiBY - 2 * Phi) / (2 * dy),
        2,
    )
    result = 0.5 * gamma * np.mean(gradient_2)
    return result


def Calc_dfdPhi(
    Phi: np.ndarray,
    Temperature: float,
    G_list: list,
    Tmax: float = 1000.0,
):
    """
    Calculates the Derivative of the Gibbs energy with respect to the order
    parameter

    :param Phi: the phase field
    :param Temperature: the temperature (in C)
    :param G_list: the list of G values (in which unit?)
    :param Tmax: The maximum temperature (in C)
    :return: the derivative of the Gibbs energy with respect to the order
    parameter
    """
    G1 = G_list[0] * (Temperature / Tmax)
    G2 = G_list[1] - G1
    c = 50
    d = 0
    # e = G1  this is not used in force calculation
    a = -2 * c - 3 * (G2 - G1 - c)
    b = G2 - G1 - c - a
    # f(Phi) = a*Phi**4 + b*Phi**3 + c*Phi**2 + d*x + e
    return 4 * a * Phi ** 3 + 3 * b * Phi ** 2 + 2 * c * Phi + d


def calc_mobility(
    mobility_type: str,
    mu_0: float,
    Temperature: float,
    T_max: float,
    T_min: float,
):
    if mobility_type == 'constant':
        # the mobility is always set to mu0
        Mobility = mu_0
    elif mobility_type == 'linear':
        # the mobility mu = a * T + b with a and b such that:
        # mu(Tmax) / mu(Tmin) = 2
        # average mu over [Tmin, Tmax] = mu0
        tmp = 0.5 * (T_max + T_min) / (T_max - 2.0 * T_min) + 1
        b = mu_0 / tmp
        a = b / (T_max - 2.0 * T_min)
        Mobility = a * Temperature + b
        assert Mobility > 0, 'The mobility should be bigger than 0.'
    elif mobility_type == 'exp':
        # Mobility is set by c * exp(-alpha / T) such that:
        # mu(Tmax) / mu(Tmin) = 2
        # mu(Tmax) = 4/3 mu0

        alpha = np.log(2) / (1 / T_min - 1 / T_max)
        mu_max = 4.0 / 3.0 * mu_0
        # mu_min will be mu_max / 2
        c = mu_max * np.exp(alpha * (1 / T_max))
        Mobility = c * np.exp(-alpha / Temperature)
    else:
        raise NotImplementedError
    return Mobility


def Calc_Force_AC(
    Phi: np.ndarray,
    Temperature: float,
    G_list: list,
    mobility_type: str,
    T_max: float = 1000.0,
    T_min: float = 100.0,
    mu_0: float = 2.0e-4,
):
    """
    Calculating the driving force for evolution of Phi consists of
    interface + bulk contributions

    :param Phi: the phase field
    :param Temperature: the current temperature (in C)
    :param G_list: the list of Gs.
    :param mobility_type: the type of mobility which can be constant, linear,
    or exponential
    :param T_max: the maximum temperature (in C)
    :param T_min: the minimum temperature (in C)
    :param mu_0: a value for mobility which set the scale of the mobility
    (in different mobility types it is used differently).
    :return: the local deriving force for PF.
    """

    Kappa = (
        30  # Sort of interface energy, larger value,
        # more diffuse, more round shapes
    )
    Mobility = calc_mobility(
        mobility_type=mobility_type,
        mu_0=mu_0,
        Temperature=Temperature,
        T_max=T_max,
        T_min=T_min,
    )
    return Mobility * (
        2 * Kappa * Calc_Del2(Phi)
        - Calc_dfdPhi(
            Phi,
            Temperature,
            G_list,
            T_max,
        )
    )


def Update_PF(
    Phi: np.ndarray,
    Temperature: float,
    NSteps: int,
    G_list: list,
    mobility_type: str,
):
    """
    Updates the phase field.

    :param Phi: the current phase-field
    :param Temperature: Temperature (in C)
    :param G_list: the G values
    :param NSteps: the number of steps to update the phase field
    :param mobility_type: the type of mobility
    :return: the updated phase field.
    """
    dt = 1
    for Step in range(0, NSteps):
        f = Calc_Force_AC(Phi, Temperature, G_list, mobility_type)
        Phi = Phi + f * dt
    # non-periodic condition is forced here:
    Phi[:, 0] = 0
    Phi[:, -1] = 0
    Phi[0, :] = 0
    Phi[-1, :] = 0
    if np.min(Phi) < 0:
        print('Something wrong with Phi, min is negative', np.min(Phi))

    if np.max(Phi) > 1:
        print('Something wrong with Phi, max is larger than 1', np.max(Phi))

    assert np.min(Phi) >= 0, 'sth went wrong in Phi Update' + str(np.min(Phi))
    assert np.max(Phi) <= 1, 'sth went wrong in Phi Update' + str(np.max(Phi))
    return Phi, f * dt, Calc_Interface_Energy(Phi, 1.0)
