# A Furnace Environment
A furnace environment compatible with OpenAI Gym is developed here.

## Introduction to ```Furnace```

The environment designed to experiment with heat treatment in material science. The physics behind the environment is a binary Allen-Cahn phase field model 
  


     Episode Termination: 
     
### State Space
The state includes:
* Timestep: a scalar indicating the number of steps up to now,
* Temperature: the current temperature, 
* Phase Field: 2D field.

Starting State: timestep is set to zero, temperature is set at the middle of the admissible range, and the PF at each point is set randomly to a value around 0.5 with a given tolerance.
              
### Action Space
The actions are for temperature regulation or process termination:
* Action 0: reduce the temperature 
* Action 1: no change the temperature 
* Action 2: increase the temperature 
* Action 3: stop the process 
  
### Reward
The reward at each step show the change between the "similarity" measure of the previous and current state. The similarity measure quantifies the similarity of the current PF with the target PF (which is a circular domain of phase 1 with the inputed total volume fraction). Both PFs are treated as images and the similarity is a modification of Intersection over Union (IoU).

### Termination
Either: 
* Reach the maximum number of allowed steps, 
* The change in dphi is smaller than a value (if the give value is 0.0 this condition is effectively ignored). 
* The temperature is out of range; this condition is active only if termination_temperature_criterion = True. In the case where the parameter is False the temperature is set to the corresponding boundary value if it gets out of bounds. 
* If the action 3 is chosen.

## How to use it?
### Installation 
* First create a new virtual environment or activate an old one with python3.9 (here we create a new one):
```commandline
conda create -n furnenv python=3.9
conda activate furnenv
```
* Update your pip,
```commandline
pip install --upgrade pip
```
* Clone the repo and enter the code's directory:
```commandline
git clone git@github.com:nima-siboni/furnace_env.git
cd furnace_env
```
* Install the requirements:
```commandline
pip install -r requirements.txt
```

* Finally install the "Furnace" package:
```commandline
pip install -e .
```
### Create a new environment
After installation, you can create an instance of the furnace environment:
* with default parameters by
```python
from Furnace import Furnace
env = Furnace()
env.reset()
```
* loading the environment config variables from a cfg file:
```python
import json  # for reading the config dict from a config file.
from Furnace import Furnace
env_config = json.load(open('./env_config.cfg'))
env = Furnace(env_config=env_config)
env.reset()
```
* with custom values for the parameters as a dictionary:
```python
from Furnace import Furnace
env_config = {"N": 2100,
             "L": 128,
             "minimum temperature": 100,
             "maximum temperature": 1000,
             "desired_volume_fraction": 0.2,
             "temperature change per step": 60,
             "number of PF updates per step": 100,
             "gamma": 1000,
             "termination_change_criterion": 0,
             "termination_temperature_criterion": "False",
             "mobility_type": "exp",
             "G_list": "1.0, 1.0",
             "shift_PF": -0.5,
             "initial_PF_variation": 0.01,
             "stop_action": "True",
             "energy_cost_per_step": 0.0,
             "verbose": "False"
             }
env = Furnace(env_config=env_config)
env.reset()
```

## Configuring ```Furnace```
tbd.
