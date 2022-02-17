
# This is a module which contains the templates for the classes to define MDP problems to feed to active inference
# This is the pool of actions and states that we can manipulate
import numpy as np

class MDPIsAt:
    def __init__(self): 
        self.state_name = 'isAt'    # This is the name of the state
        self.action_name = ['Idle', 'moveTo']

        self.V = np.array([0, 1])  # Allowable policies, it indicates policies of depth 1
        self.B = np.zeros((2, 2, 2))  # Allowable actions initiation
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)  # Idle action
        self.B[:, :, 1] = np.array([[1, 1],  # move(loc): a_moveBase makes isAt true
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = np.zeros((2, 2, 2)) - 1  # No preconditions needed for Idle and a_mv, set to -1

        # Likelihood matrix matrices
        # ----------------------------------------------------------
        self.A = np.eye(2)  # Identity mapping
        # Prior preferences, initially set to zero, so no preference
        # -----------------------------------------------------------
        self.C = np.array([[0.], [0.]])
        # Belief about initial state, D
        # -----------------------------------------------------------
        self.D = np.array([[0.5], [0.5]])
        # Initial guess about the states d, all equally possible, this is updated over time
        # -----------------------------------------------------------
        self.d = np.array([[0.5], [0.5]])

        # Preference about actions, idle is slightly preferred
        # -----------------------------------------------------------
        self.E = np.array([[1.01], [1]])
        # Learning rate for initial state update
        # -----------------------------------------------------------
        self.kappa_d = 0.2

    # Default habits
    def set_default_preferences(self):
        self.E = np.array([[1.01], [1]])

    # Default initial estimate
    def reset_belief(self):
        self.d = np.array([[0.5], [0.5]])