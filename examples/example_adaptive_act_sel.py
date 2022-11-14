#!/usr/bin/env python3

# Simple example to create an AI agent which selects actions to satisfy task needs

import numpy as np
from decision_making import ai_agent, state_action_templates, adaptive_action_selection, parallel_action_selection

## Initialization
# ----------------- 
# Define the required mdp structures from the templates
mdp_isAt = state_action_templates.MDPIsAt() 
mdp_isHolding = state_action_templates.MDPIsHolding() 
mdp_isReachable = state_action_templates.MDPIsReachable() 
mdp_isPlacedAt = state_action_templates.MDPIsPlacedAt() 
mdp_isVisible = state_action_templates.MDPIsVisible() 

# Agent with following states [isAt, isHolding, isReachable, isPlacedAt, isVisible]
ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_isHolding), ai_agent.AiAgent(mdp_isReachable), ai_agent.AiAgent(mdp_isPlacedAt), ai_agent.AiAgent(mdp_isVisible)]
ai_agent_task[3].set_preferences(np.array([[1.], [0.]]))

# Loop for the execution of the task, ideally this will be given by the tick of a BT
for i in range(25):
    # Set preference for a particular state to be achieved. Follow indexing in ai_agent_task [isAt, isHolding, isReachable, isPlacedAt, isVisible]
    # Get the parameters of the object to be holding to define a suitable observation
    # most likely from the parameter server (TODO add the parameters from the BT)

    # Set the observation from the current readings, (TODO) the logic of the observations need to be specified for the task and the parameters passed by the BT
    # in terms of products. When an observation is unavailable set it to 'null'
    if i < 5:
        obs = ['null', 1, 1, 1, 1]
    if i>= 5 and i < 10:
        obs = ['null', 1, 0, 1, 1]
    if i>= 10 and i < 15:
        obs = ['null', 1, 0, 1, 0]    
    if i>= 15 and i < 20:
        obs = ['null', 0, 0, 1, 0]  
    if i>= 20:
        obs = ['null', 1, 0, 0, 0]  
    # To test parallel adaptive action selection swap commented lines 41 and 40, or see 'example_parallel_act_sel.py'
    outcome, curr_acti = adaptive_action_selection.adapt_act_sel(ai_agent_task, obs)
    # outcome, curr_acti = parallel_action_selection.par_act_sel(ai_agent_task, obs)
    print(outcome)
    print('current action', curr_acti)
