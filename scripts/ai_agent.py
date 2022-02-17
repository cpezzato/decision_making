## Agent class
# This script contains the active inference agent class. 
# This is ageneral class that uses the aip module which allows to update the mdp structure given an observation and a preference. 
# Initialized using the mdp templates 

import aip 
import numpy as np
import copy

class AiAgent(object):
    def __init__(self, mdp):
        self._mdp =  mdp    # This contains the mdp structure for the active inference angent

        # Initialization of variables
        self.n_policies = np.shape(self._mdp.V)[0]  # Number of allowable policies
        self.n_states = np.shape(self._mdp.B)[0]  # Number of states
        self.n_actions = np.shape(self._mdp.B)[2]  # Number of controls
        self.n_outcomes = self.n_states  # Number of sensory inputs, same as the states
        self.n_iter = 4  # Varitional bayes iterations
        self.t_horizon = 2  # Time horizon to look one step ahead

        # Assigning local variables to this instance of the function
        # ------------------------------------------------------------------------------------------------------------------
        self.policy_indexes_v = self._mdp.V  # Indexes of possible policies
        self.policy_post_u = np.zeros([self.n_policies, self.t_horizon])  # Initialize vector to contain posterior probabilities of actions

        # Prior expectation over hidden states at the beginning of the trial
        if hasattr(self._mdp, 'd'):
            self._mdp.D = self.aip_norm(self._mdp.d)
        elif hasattr(self._mdp, 'D'):
            self._mdp.D = self.aip_norm(self._mdp.D)
        else:
            self._mdp.D = self.aip_norm(np.ones((self.n_states, 1)))

        # Likelihood matrix
        self.likelihood_A = self.aip_norm(self._mdp.A)

        # Transition matrix
        self.fwd_trans_B = np.zeros((self.n_states, self.n_states, self.n_actions))
        self.bwd_trans_B = np.zeros((self.n_states, self.n_states, self.n_actions))

        for action in range(self.n_actions):
            # Retrieve forward messages, B
            self.fwd_trans_B[:, :, action] = self.aip_norm(self._mdp.B[:, :, action])
            # Retrieve backward messages, transpose of B
            self.bwd_trans_B[:, :, action] = np.transpose(self.aip_norm(self._mdp.B[:, :, action]))

        # Prior preferences (log probabilities) : C
        self.prior_C = self.aip_log(self.aip_softmax(copy.copy(self._mdp.C)))
        # Preferences over policies
        self.prior_E = self.aip_log(self.aip_norm(self._mdp.E))

        # Current observation
        # ------------------------------------------------------------------------------------------------------------------
        self.outcome_o = np.zeros([1, self.t_horizon]) - 1
        # If outcomes have been specified then set it, otherwise leave to 0
        if hasattr(mdp, 'o'):
            self.outcome_o[0, 0] = self._mdp.o  # Outcomes here are indicated in 'compact notation' with 1 and 2
        # Putting observations in sparse form, initialization
        self.sparse_O = np.zeros((1, self.n_states, self.n_outcomes))  # Outcomes here are indicated as [1 0], [0 1]

        # Posterior states
        # ------------------------------------------------------------------------------------------------------------------
        # Initial guess about posterior hidden states, in 'compact notation' with 1 and 2
        self.hidden_states_s = np.zeros([1, self.t_horizon]) - 1  # Unassigned states and values are -1
        self.hidden_states_s[0, 0] = np.argmax(self._mdp.D)  # Get index of max value and set as initial state

        # Initialize posterior expectation over hidden states
        self.post_x = np.zeros([self.n_states, self.t_horizon, self.n_policies]) + 1 / self.n_states
        self.sparse_post_X = np.zeros([self.n_states, self.t_horizon])
        self.sparse_post_X[:, 0] = np.transpose(self._mdp.D)
        # Set the current state to what contained in D. At the next step it is still uncertain, so we leave it as that
        for policy in range(self.n_policies):
            self.post_x[:, 0, policy] = np.transpose(self._mdp.D)

    def infer_states(self, obs):
        # This method akes as argument
        pass
       
    # Update observations for an agent
    def set_observation(self, obs):
        self._mdp.o = obs
    
    # Update the preferences of the agent over the states it cares about
    def set_preferences(self, pref):
        self._mdp.C = pref

    # Minimize the free energy for joint state estimation and action
    def minimize_f(self):
        # Update the mdp from the class constructor whenever we are required to perform action selection
        self._mdp = aip.aip_select_action(self._mdp)
        return self._mdp

    # Get current action
    def get_action(self):
        return self._mdp.u[0]

    # Get current best estimate of the state
    def get_current_state(self):
        return self._mdp.s

    # Get current best estimate of the state
    def get_d(self):
        return self._mdp.d

    # Active inference routines

    def infer_policies():
        pass

    def aip_log(self, var):
        # Natural logarithm of an element, preventing 0. The element can be a scalar, vector or matrix
        return np.log(var + 1e-16)

    def aip_norm(self, var):
        # Normalisation of probability matrix (column elements sum to 1)
        # The function goes column by column and it normalise such that the
        # elements of each column sum to 1
        # In case of a matrix
        for column_id in range(np.shape(var)[1]):  # Loop over the number of columns
            sum_column = np.sum(var[:, column_id])
            if sum_column > 0:
                var[:, column_id] = var[:, column_id] / sum_column  # Divide by the sum of the column
            else:
                var[:, column_id] = 1 / np.shape(var)[0]  # Divide by the number of rows
        return var

    def aip_softmax(self, var):
        # Function to compute the softmax of a given column array: sigma = exp(x) / sum(exp(x))
        ex = np.exp(var)  # Compute exponential
        for i in range(np.shape(var)[0]):
            var[i] = ex[i] / np.sum(ex)  # Compute softmax element by element
        return var