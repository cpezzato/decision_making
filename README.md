# Repository for decision making through active infrence

This repo contains discrete active inference for combining external goals and internal system requirements. The AiAgent class implements the methods for active inference while adaptive_action_selection implements the algorithm for conflicts resolution. The method parallel_action_selection instead, provides a list of possible alternative actions. Some lists contains multiple actions, which can then be executed in parallel. Every list in the list of possible actions is a different strategy.

The repo contains simple examples to use the class and the adaptive action selection algorithm. 

If you find this work useful, please consider citing the connected paper: 

- *Pezzato, Corrado, Carlos Hernández Corbato, Stefan Bonhof, and Martijn Wisse. "Active inference and behavior trees for reactive action planning and execution in robotics." IEEE Transactions on Robotics 39, no. 2 (2023): 1050-1069*

## Installation
Install the package with:
````bash
pip3 install .
````

or, for allowing local changes:
##
````bash
python3 -m pip install -e .
````

### Istructions to run
Test an example script from the example folder:

````bash
python3 examples/example_parallel_act_sel.py
````

