#!/usr/bin/env python

# Simple example to create an AI agent which selects actions to satisfy internal needs

import numpy as np
import ai_agent                      
import int_req_templates               


# Function to emulate gtting an observation from the environment
def get_obs_env():
    # For the battery we have 3 possible observations  ['ok', 'low', 'critcal'] = [0, 1, 2]
    obs = 0
    return obs

## Initialization
# ----------------- 
# Define the required mdp structures 
mdp_battery = int_req_templates.MDPBattery() 

# Define ai agent with related mdp structure to reason about
ai_agent_internal = ai_agent.AiAgent(mdp_battery)

## Decision making
#-------------------
# A typical sequence for decision making, ideally this should be repeated at a certain frequency

# Set the preference for the battery 
ai_agent_internal.set_preferences(np.array([[1.], [0], [0]])) # Fixed preference for battery ok, following ['ok', 'low', 'critcal'] 

for i in range(10):
    # Set the observation from the environment
    if i < 9:
        ai_agent_internal.set_observation(0)
    else:
        ai_agent_internal.set_observation(2)
    
    # Minimize free energy to get state update and action
    mdp_battery = ai_agent_internal.minimize_f()

    # Printouts
    print('The battery state is:',  ai_agent_internal._mdp.state_names[ai_agent_internal.get_current_state()])
    print('The selected action is:', ai_agent_internal._mdp.action_names[int(ai_agent_internal.get_action())])

    # Belief after first run
    print('Belief about battery state', ai_agent_internal._mdp.d)

    ## TODO now observatoins have a direct influence on the action selection because of the way aip was used in spm. However, I should use one observation to update the state and then run the action selection with the updated state. See pseudocode paper
    # This is necessary to avoid jumping back and forth. The ones for action selection do not have d and o but only D.