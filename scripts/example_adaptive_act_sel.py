#!/usr/bin/env python

# Simple example to create an AI agent which selects actions to satisfy task needs

import numpy as np
import ai_agent                      
import state_action_templates               

def adaptive_action_selection(agent, obs):
    # This function computes the next best action based on the provided mdp structures. It checks for current desired states and runs an active inference loop for the ones with
    # an active preference. When an action is selected, its preconditions are checked looking at the estimatd states in the mdp structures. If they are met the action is selected 
    # to be executed, if not, the loop is repeted with pushed high priority preconditions. If no action is found the algorithm returns failure. 
    
    #  At each new iteration (or tick from the BT), restore all available actions and remove high priority priors that are already satisfied
    n_mdps = len(agent)
    for i in range(n_mdps):
        agent[i].reset_habits()
        for index in range(len(agent[i]._mdp.C)):  # Loop over values in the prior
            if agent[i]._mdp.C[index] > 0 and index == np.argmax(agent[i].get_current_state()):
                # Remove precondition pushed since it has been met
                # agent[i].set_preferences(0, index)
                pass

    # Initialize actions and current states list for this adaptive selection process
    action_found = 0
    u = [-1]*n_mdps
    current_states = ['null']*n_mdps
    looking_for_alternatives = 0

    while action_found == 0:
        for i in range(n_mdps):
            # Compute free energy and posterior states for each policy if an observation is vailable
            if obs[i] >= 0:
                if not looking_for_alternatives:
                    agent[i].infer_states(obs[i])
                # Compute expected free-energy and posterior over policies
                G, u[i] = agent[i].infer_policies()
                current_states[i] = agent[i]._mdp.state_names[np.argmax(agent[i].get_current_state())]
                #print('Actions found:', u[i])
        # If all the actions are idle, we can return success since no action is required
        if np.max(u) == 0:
            if not looking_for_alternatives:
                print("No action needed")
                outcome = 'success'
                curr_action = 'idle'
            else:
                print("No action found for this situation")
                outcome = 'failure'
                curr_action = 'idle'
            break   # Exit the while loop
        # Else, we check the preconditions of the selected action, push missing states, and re-run the action selection
        else:
            for i in range(n_mdps):
                # Get preconditions to be satisfied for this action if it is not idle
                if u[i] > 0:
                    prec = agent[i]._mdp.preconditions[u[i]]
                    # Flag for unmet preconditions
                    _unmet_prec = 0
                    # Check if the preconitions are satisfied and if not add preference with high priority on respective priors 
                    for item in range(len(prec)):
                        if (prec[item] not in current_states) and (prec[item]!='none'):
                            _unmet_prec = 1
                            looking_for_alternatives = 1
                            print('There are unmet preconditions for action', agent[i]._mdp.action_names[u[i]])
                            # Get index of missing state and push a prior on that state
                            for j in range(n_mdps):
                                if prec[item] in agent[j]._mdp.state_names:
                                    agent[j].set_preferences(2, agent[j]._mdp.state_names.index(prec[item]))  # (value, index)
                            # Inhibit current action for the inner adaptation loop since missing preconditions
                            agent[i].reset_habits(u[i])
                        # If the preconditions are met after checking we can execute the action
                    if _unmet_prec == 0:
                        print("Action found:", agent[i]._mdp.action_names[u[i]])
                        action_found = 1
                        outcome = 'running'
                        curr_action = 'some_action'
                        break   # Exit the while loop

    #print(current_states)

    curr_action = 'idle'
    outcome = 'success'
    return outcome, curr_action

## Initialization
# ----------------- 
# Define the required mdp structures from the templates
mdp_isAt = state_action_templates.MDPIsAt() 
mdp_isHolding = state_action_templates.MDPIsHolding() 
mdp_isReachable = state_action_templates.MDPIsReachable() 
mdp_isPlacedAt = state_action_templates.MDPIsPlacedAt() 

# Agent with following states [isAt, isHolding, isReachable, isPlacedAt]
ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_isHolding), ai_agent.AiAgent(mdp_isReachable), ai_agent.AiAgent(mdp_isPlacedAt)]

# Loop for the execution of the task, ideally this will be given by the tick of a BT
for i in range(1):
    # Set preference for a particular state to be achieved: The robot is holding an object
    ai_agent_task[3].set_preferences(np.array([[1.], [0.]]))
    # Get the parameters of the object to be holding to define a suitable observation
    # most likely from the parameter server (TODO add the parameters from the BT)

    # Set the observation from the current readings, (TODO) the logic of the observations need to be specified for the task and the parameters passed by the BT
    # in terms of products. When an observation is unavailable
    obs = [0, 1, 1, 1]
    outcome, curr_acti = adaptive_action_selection(ai_agent_task, obs)

