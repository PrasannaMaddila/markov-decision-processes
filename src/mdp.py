"""
This module implements the MDP class, a Gym environment
that models a Markovian Decision Process.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SummableMDP(gym.Env):
    """
    Class to implement a stationary Markov Decision Process
    as a gym environment.
    """
    def __init__(self, num_actions, num_states, rewards, transitions, init_state, timesteps=100):
        """
        Initialising the MDP with `num_states` states and
        `num_actions` actions using spaces.Discrete.
        Note: Both spaces are zero-indexed !!!
        """
        # counters for determining end 
        self.timesteps = timesteps
        self.curr_time = 0
        self.observation_space = spaces.Discrete(num_states)
        self.action_space = spaces.Discrete(num_actions)
        self.rewards = rewards
        self.transitions = transitions

        if self.rewards.keys() != self.transitions.keys(): 
            raise RuntimeError("Rewards and Transitions defined on different domains")
        if init_state < 0 or init_state >= num_states:
            raise RuntimeError(f"Invalid init state{init_state}: out of bounds!")

        self.init_state = init_state    # storing this for reset()
        self.curr_state = init_state
        self.curr_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.curr_reward = 0.0
        self.curr_state = self.init_state
        self.old_state  = None
        self.old_state = None

    def step(self, action: int):
        """
        Simulate a step in an MDP. This means an action is taken,
        so that s ---> s',
        """
        self.curr_time += 1
        try: 
            probas = self.transitions[(self.curr_state, action)]
        except KeyError:
            return (self.curr_state, self.curr_reward, 
                    self.curr_time == self.timesteps,"Undefined state accessed. Breaking early.")

        # Create a mask that is 1 only where we select the element.
        # this mask has to be generated using transition probability
        # as defined in `probas`.
        mask = np.zeros(self.observation_space.n, dtype=np.int8); 
        mask[np.random.choice( range(self.observation_space.n), p=probas)] = 1

        # Selecting new state using observation space.
        self.old_state = self.curr_state
        self.curr_state = self.observation_space.sample(mask=mask)
        
        # Getting reward 
        self.curr_reward += self.rewards[(self.old_state, action)]
        
        # returning results of step
        return self.curr_state, self.curr_reward, self.curr_time == self.timesteps, None

    def sample(self):
        return self.action_space.sample()
    
if __name__ == "__main__":
    # simulating the text example.
    mdp_states, mdp_actions = 3, 2

    # rewards are represented as (curr_state,action) key-value paris
    mdp_rewards = {
        (0,0): -5, (0,1): 10,
        (1,0): 5,  (1,1): 0,
        (2,0): 20
        }

    # transition probabilities are represented as
    # (new_state, old_state, action): dict key-value pairs
    mdp_transitions = {
            (0,0): [0.9, 0, 0.1], (0,1): [0,1,0], 
            (1,0): [0,1,0], (1,1): (0.8, 0.2, 0), 
            (2,0): [0.9, 0.1, 0]
            }

    # just checking
    assert ( mdp_transitions.keys() == mdp_rewards.keys() )
    
    # initialising
    mdp = SummableMDP(
            mdp_actions, mdp_states, 
            mdp_rewards, mdp_transitions, 
            init_state = 1
            )
    
    done, undefined_acess = False, 0
    while not done:
        curr_action = mdp.sample()
        curr_state, reward, done, info = mdp.step(curr_action)
        if info is not None: 
            undefined_acess+=1 
    
    print(f"Current reward: {mdp.curr_reward}")
    print(f"Undefined state accesses: {undefined_acess}")
    mdp.close()
