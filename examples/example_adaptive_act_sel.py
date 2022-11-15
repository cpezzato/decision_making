#!/usr/bin/env python3

# Simple example to create an AI agent which selects actions to satisfy a task

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

# Agent with following states [isAt, isHolding, isReachable, isPlacedAt, isVisible], see templates
ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_isHolding), ai_agent.AiAgent(mdp_isReachable), ai_agent.AiAgent(mdp_isPlacedAt), ai_agent.AiAgent(mdp_isVisible)]
# Define the task for an agent by setting the preferences
ai_agent_task[3].set_preferences(np.array([[1.], [0.]]))

# Loop for the execution of the task, ideally this will be given by the tick of a BT
for i in range(25):
    # Set the observation from the current readings, the logic of the observations need to be specified for the task. 
    # When an observation is unavailable set it to 'null'
    
    if i < 5:
        obs = ['null', 1, 1, 1, 1]
    if i>= 5 and i < 10:
        obs = ['null', 1, 0, 1, 1]
    if i>= 10 and i < 15:
        obs = ['null', 1, 0, 1, 0]    
    if i>= 15 and i < 20:
        obs = ['null', 0, 0, 1, 0]  
    if i>= 20:
        obs = ['null', 0, 0, 0, 0]  

    # To test parallel adaptive action selection swap commented lines below
    outcome, curr_acti = adaptive_action_selection.adapt_act_sel(ai_agent_task, obs)
    #outcome, curr_acti = parallel_action_selection.par_act_sel(ai_agent_task, obs)
   
    print('Status:', outcome)
    print('Current action(s):', curr_acti)
