[![Python application](https://github.com/nima-siboni/Furnace-Env/actions/workflows/python-app.yml/badge.svg)](https://github.com/nima-siboni/Furnace-Env/actions/workflows/python-app.yml)
# A Furnace Environment
A furnace environment compatible with Gymnasium is developed here.

## Introduction to ```Furnace```

The environment is designed to experiment with heat treatment in material science. The physics behind the environment is a 2D binary Allen-Cahn phase field model.

### State Space
The state includes:
* Timestep: scaled number of taken steps where the scaling factor is `horizon`,
* Temperature: the linearly transformed temperature, such that it is 0 at `minimum_temperature` and 1 at `maximum_temperature`,
* Phase Field: 2D field, where the values are by definition between 0 and 1.
Starting State: timestep is set to zero, temperature is set at the middle of the admissible range, and the PF at each point is set randomly to a value around 0.5 with a given tolerance.

![](./statics/sample.png)
### Action Space
The actions are for temperature regulation or process termination:

* Action 0: reduce the temperature,
* Action 1: no change the temperature,
* Action 2: increase the temperature,
* Action 3: stop the process

The changes in the temperate are determined by the parameter `temperature_change_per_step` which is in Celsius. The temperature is bounded between `minimum_temperature` and `maximum_temperature`.
### Reward
The reward at each step has two components:
* one component is related to the energy cost of the process,
* the other component is related to how close we are to the desired microstructure.

The first contribution is easy to calculate: we approximate the cost of running the furnace to be proportional to
the difference between the temperature of the furnace and the ambient temperature. The proportionality factor is `energy_cost_per_step` which is in units of $/s/C.

Calculation of the second contribution is a bit more evolved. We first define a quantity called "similarity";
it measures the similarity  the current PF with the target PF (which is a circular domain of phase 1 which covers `desired_volume_fraction` of the whole box).
Using this similarity, we define a reward function to be the change in the "similarity" (to the target image) between the previous and the current state.

The similarity between two images are defined as the correlation between the absolute values of the
fourier transforms of the two images:

$E=mc^2$


### Termination
The process is terminated under the following conditions which are all configurable (see configuration subsection):

* Reach the maximum number of allowed steps,
* The change in dphi is smaller than a value (if the give value is 0.0 this condition is effectively ignored).
* The temperature is out of range; this condition is active only if termination_temperature_criterion = True. In the case where the parameter is False the temperature is set to the corresponding boundary value if it gets out of bounds.
* If the action 3 is chosen.

## How to use it?
### Installation
* Update your pip,
```commandline
pip install --upgrade pip
```
* Clone the repo and enter the code's directory:
```commandline
git clone git@github.com:nima-siboni/furnace_env.git
cd furnace_env
```
* create a new virtual environment or activate an old one with python3.9 (here we create a new one):
```commandline
conda env create -f environment.yaml
```
* Finally install the "Furnace" package:
```commandline
pip install -e .
```
### Create a new environment
After installation, you can create an instance of the furnace environment:
* with default parameters by

```python
from furnace import Furnace

env = Furnace()
env.reset()
```
* loading the environment config variables from a cfg file:

```python
import json  # for reading the config dict from a config file.
from furnace import Furnace

env_config = json.load(open('./env_config.cfg'))
env = Furnace(env_config=env_config)
env.reset()
```
* with custom values for the parameters as a dictionary:

```python
from furnace import Furnace

env_config = {"horizon": 2100,
              "dimension": 120,
              "minimum_temperature": 100,
              "maximum_temperature": 1000,
              "desired_volume_fraction": 0.2,
              "temperature_change_per_step": 60,
              "number_of_pf_updates_per_step": 100,
              "gamma": 1000,
              "termination_change_criterion": 0.0,
              "use_termination_temperature_criterion": False,
              "mobility_type": "exp",
              "g_list": [1.0, 1.0],
              "shift_pf": -0.5,
              "initial_pf_variation": 0.01,
              "use_stop_action": True,
              "energy_cost_per_step": 0.0,
              "verbose": False}
env = Furnace(env_config=env_config)
env.reset()
```

## Configuring ```Furnace```
Important configuration parameters for the environment are the followings:

* ```N```: the maximum number of steps
* ```L```: the spatial dimension of the domain
* ```minimum temperature```: minimum temperature (in C),
* ```maximum temperature```: maximum temperature (in C),
* ```desired_volume_fraction```: the target volume fraction of phase 1; the target PF is a circle with this volume fraction,
* ```temperature change per step```: the temperature change in case of actions 0 and 2,
* ```number of PF updates per step```: the number of PF updates for each environment step,
* ```termination_change_criterion```: if the (absolute) PF change is smaller than this value the process is terminated; to disable  this condition set it to zero,
* ```termination_temperature_criterion```: whether to terminate the process when the agent brings the temperature out of the [Tmin, Tmax] range; setting it to False keeps the temperature at the boundary value and does not terminate the process,
* ```stop_action```: whether to have the termination action or not,
* ```energy_cost_per_step```: a factor which is multiplied by the difference between the temperature of the furnace and the temperature of the ambient; the larger this value the more expensive would be to run the furnace at higher temperatures; to remove the energy cost from optimization set this value to 0.
* ```mobility_type```: "const", "exp", or "linear"; determines how the mobility changes with temperature
* ```gamma```: a factor which is multiplied by the interface energy (just for reporting),
* ```initial_PF_variation```: the variation of the average PF around 0.5 at reset,
* ```G_list```: values related to the relative stability of different phases, e.g. "1.0, 1.0",
* ```shift_PF```: the initial shift of PF values, for easier learning set it to -0.5,

Note that logical conditions and lists are inputed as strings with double quatations, like "False" (not False), and "1.0, 2.0" (not [1.0, 2.0]). Sooorrry!
