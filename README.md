# A Furnace Environment
A furnace environment compatible with OpenAI Gym is developed here.

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
* loading the environment config variables from a cfg file:
```python
import json  # for reading the config
from Furnace import Furnace
env_config = json.load(open('./env_config.cfg'))
env = Furnace(env_config=env_config)
env.reset()
```
