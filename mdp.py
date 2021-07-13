import pdb
from dist import uniform_dist, delta_dist, mixture_dist
from util import *
import random
import numpy as np

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]

class MDP:
    # Needs the following attributes:
    # states: list or set of states
    # actions: list or set of actions
    # discount_factor: real, greater than 0, less than or equal to 1
    # start: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    # These are functions:
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward

    def __init__(self, states, actions, transition_model, reward_fn, 
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

# Perform value iteration on an MDP, also given an instance of a q
# function.  Terminate when the max-norm distance between two
# successive value function estimates is less than eps.
# interactive_fn is an optional function that takes the q function as
# argument; if it is not None, it will be called once per iteration,
# for visuzalization

# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values This must be
# initialized before interactive_fn is called the first time.

def value_iteration(mdp, q, eps = 0.01, interactive_fn = None,
                    max_iters = 10000):
    # Your code here
    #series = q.series
    #actions = q.actions
    gamma = mdp.discount_factor
    R = mdp.reward_fn
    T = mdp.transition_model

    #Assumes that Q has already been initialised
    while True:

        q_new = q.copy()
        for s in q.states:
            for a in q.actions:
                expected_reward = 0
                for s_ in q.states:
                    #expected_reward += T(s,a).prob(s_) * R(s_,greedy(q,s_))
                    max_q_a = max([q.get(s_,a) for a in q.actions])
                    expected_reward += T(s,a).prob(s_) * max_q_a
                new_value = R(s,a) +gamma*expected_reward
                q_new.set(s,a,new_value)

        #find the maximum difference between old and new values
        diffs = [abs(q.get(s,a)-q_new.get(s,a)) for s in q.states for a in q.actions]
        if max(diffs) < eps:
            return q_new
        else:
            q = q_new.copy()

# Compute the q value of action a in state s with horizon h, using
# expectimax
def q_em(mdp, s, a, h):
    # Your code here
    gamma = mdp.discount_factor
    R = mdp.reward_fn
    T = mdp.transition_model

    #base cases
    if h==0: 
        return 0
    elif h==1: 
        return R(s,a)
    else:
        sum_term = sum([T(s, a).prob(s_)* q_em(mdp, s_, a, h-1) for s_ in T(s,a).support()])
        return R(s,a) + gamma*sum_term


# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    """ Return Q*(s,a) based on current Q

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q_star = value(q,0)
    >>> q_star
    10
    """
    # Your code here
    return max(q.get(s,a)for a in q.actions)


q = TabularQ([0,1,2,3],['b','c'])
q.set(0, 'b', 5)
q.set(0, 'c', 10)
print(value(q,0))

# Given a state, return the action that is greedy with reespect to the
# current definition of the q function
def greedy(q, s):
    """ Return pi*(s) based on a greedy strategy.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> greedy(q, 0)
    'c'
    >>> greedy(q, 1)
    'b'
    """
    # Your code here
    rewards = [q.get(s,a)for a in q.actions]
    return q.actions[np.argmax(rewards)]

q = TabularQ([0,1,2,3],['b','c'])
q.set(0, 'b', 5)
q.set(0, 'c', 10)
q.set(1, 'b', 2)
print(greedy(q, 0))
print(greedy(q, 1))


def epsilon_greedy(q, s, eps = 0.5):
    """ Return an action.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> eps = 0.
    >>> epsilon_greedy(q, 0, eps) #greedy
    'c'
    >>> epsilon_greedy(q, 1, eps) #greedy
    'b'
    """
    if random.random() < eps:  # True with prob eps, random action
        # Your code here
        return uniform_dist(q.actions).draw()
    else:
        return greedy(q, s)
        


