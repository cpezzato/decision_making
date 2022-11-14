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

ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_isBlockAt), ai_agent.AiAgent(mdp_isLocFree)]

start_time = time.time()
ai_agent_task[1].set_preferences(np.array([[1.], [0.]]))

# Loop for the execution of the task, ideally this will be given by the tick of a BT
for i in range(15):
    if i < 5:
        obs = [1, 1, 1]
    if i>= 5 and i < 10:
        obs = [1, 1, 0]
    if i>= 10 and i < 15:
        obs = [1, 0, 0]    
    # outcome, curr_acti = adaptive_action_selection.adapt_act_sel(ai_agent_task, obs)
    outcome, curr_plan = parallel_action_selection.par_act_sel(ai_agent_task, obs)
    print(outcome)
    print('Current plan', curr_plan)
#print("--- %s seconds ---" % (time.time() - start_time))