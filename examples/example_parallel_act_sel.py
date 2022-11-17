#!/usr/bin/env python3

# Simple example to create an AI agent which selects actions to satisfy task needs

import numpy as np
from decision_making import ai_agent, state_act_point_robot, parallel_action_selection
import time

## Initialization
# ----------------- 
# Define the required mdp structures from the templates
mdp_isAt = state_act_point_robot.MDPIsAt() 
mdp_isBlockAt = state_act_point_robot.MDPIsBlockAt() 
mdp_isLocFree = state_act_point_robot.MDPIsLocFree() 
mdp_isCloseTo = state_act_point_robot.MDPIsCloseTo() 

ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_isBlockAt), ai_agent.AiAgent(mdp_isLocFree), ai_agent.AiAgent(mdp_isCloseTo)]

start_time = time.time()
ai_agent_task[1].set_preferences(np.array([[1.], [0.]]))

# Loop for the execution of the task, ideally this will be given by the tick of a BT
for i in range(15):
    if i < 5:
        obs = ['null', 1, 0, 1]
    if i>= 5 and i < 10:
        obs = ['null', 1, 0, 0]
    if i>= 10 and i < 15:
        obs = ['null', 1, 0, 0]    

    outcome, curr_plan = parallel_action_selection.par_act_sel(ai_agent_task, obs)
    print(outcome)
    print('Current plan', curr_plan)
