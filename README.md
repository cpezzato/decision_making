# Repository for decision making through active infrence

This repo contains discrete active inference for combining external goals and internal system requirements. The decision making is in terms of the costs to be minimized at the current time, to be provided to the MPPI motion planner. 

## Current status
The repo is in a development stage. 

## Installation

### Requirements
You have IsaacGym installed. See https://developer.nvidia.com/isaac-gym

### Istructions ti run
Just clone the repo in a location in you computer. If you have a Conda environment for IsaacGym, activate the environment

````bash
cd <you_isaac_gym_folder>
conda acivate <env_name>
````

Then you are ready to test an example script:

````bash
cd <path/to/decision_making/scripts>
python example_battery_isaac.py
````

