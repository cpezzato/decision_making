# Module for adaptive action selection
import numpy as np

def adapt_act_sel(agent, obs):
    # This function computes the next best action based on the provided mdp structures. It checks for current desired states and runs an active inference loop for the ones with
    # an active preference. When an action is selected, its preconditions are checked looking at the estimatd states in the mdp structures. If they are met the action is selected 
    # to be executed, if not, the loop is repeted with pushed high priority preconditions. If no action is found the algorithm returns failure. 
    
    #  At each new iteration (or tick from the BT), restore all available actions and remove high priority priors that are already satisfied
    if type(agent) == list:
        n_mdps = len(agent)
    else:
        # Put it in a list form if only one mdp is provided and relative observation
        n_mdps = 1
        agent = [agent]
        obs = [obs]
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
            if obs[i] != 'null':
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
                curr_action = 'idle_success'
            else:
                print("No action found for this situation")
                outcome = 'failure'
                curr_action = 'idle_fail'
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
                        curr_action = agent[i]._mdp.action_names[u[i]]
                        break   # Exit the while loop
    return outcome, curr_action
