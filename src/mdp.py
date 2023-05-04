"""
This module implements the MDP class, a Gym environment
that models a Markovian Decision Process.
"""

import numpy as np
import gymnasium as gym
import multiprocessing as mp
from gymnasium import spaces

class MDP(gym.Env):
    """
    Class to implement a stationary Markov Decision Process
    as a gym environment.
    """

    def __init__(
        self,
        num_actions: int,
        num_states: int,
        rewards: dict[(int, int):float],  # r(s,a) \in R
        transitions: dict[(int, int):int],  # (s,a) ~> s'
        init_state: int,  # initial state
        beta: float,  # discount factor
        f: list[tuple[int]],  # strategy used by controller
        timesteps: int = 100,
    ):
        """
        Initialising the MDP with `num_states` states and
        `num_actions` actions using spaces.Discrete.
        Note: Both spaces are zero-indexed !!!
        """
        # counters for determining end of run
        self.timesteps, self.curr_time = timesteps, 0

        # main variables
        self.observation_space = spaces.Discrete(num_states)
        self.action_space = spaces.Discrete(num_actions)
        self.rewards = rewards
        self.transitions = transitions
        self.curr_reward = 0.0
        self.f = f
        self.beta = beta

        if self.rewards.keys() != self.transitions.keys():
            raise RuntimeError("Rewards and Transitions defined on different domains")
        if init_state < 0 or init_state >= num_states:
            raise RuntimeError(f"Invalid init state{init_state}: out of bounds!")

        # storing state-related variables
        self.init_state = init_state  # storing this for reset()
        self.curr_state = init_state

        # calculate the transition matrix
        # using the strategy and the transition probas
        self._calculate_P()

    def reset(self, seed=None, options=None):
        """
        Reset the MDP to run the next trajectory
        """
        super().reset(seed=seed)
        self.curr_reward = 0.0
        self.curr_state = self.init_state
        self.old_state = None
        self.curr_time = 0

    def step(self, action: int):
        """
        Simulate a step in an MDP. This means an action is taken,
        so that s ---> s',
        """
        try:
            self.transitions[(self.curr_state, action)]
        except KeyError:
            # Using info parameter to return status
            return (
                self.curr_state,
                self.curr_reward,
                self.curr_time == self.timesteps,
                "Invalid state accessed. Not stepping.",
            )
        # state with defined transitions, stepping.
        probas = self.P[self.curr_state, :]
        self.curr_reward += self._get_reward(action)
        self.curr_state = self._get_state(probas)    # old_state <- curr_state, curr_state <- new_state
        self.curr_time += 1
        # returning results of step
        return self.curr_state, self.curr_reward, self.curr_time == self.timesteps, None

    def _calculate_P(self):
        """
        Helper function to calculate the transition matrix P
        as defined by the Markov Decision Process. This relies on
        the object being well-initialised first.
        """
        self.P = np.zeros((self.observation_space.n, self.observation_space.n))
        for i in range(self.observation_space.n):
            local_f = self.f[i]
            for j in range(self.observation_space.n):
                for k in range(len(local_f)):
                    # this length is similar to number
                    # of actions for this state.
                    self.P[i, j] += local_f[k] * self.transitions[(i, k)][j]

    def _get_state(self, probas):
        """
        Helper function to return the current state,
        using the probability distribution over each
        state in self.observation_space
        """
        # Create a mask that is 1 only where we select the element.
        # this mask has to be generated using transition probability
        # as defined in `probas`.
        mask = np.zeros(self.observation_space.n, dtype=np.int8)
        mask[np.random.choice(range(self.observation_space.n), p=probas)] = 1

        # Saving current state and selecting new state using observation space
        self.old_state = self.curr_state
        return self.observation_space.sample(mask=mask)

    def _get_reward(self, action):
        """
        Helper function to calculate the reward for each
        timestep, using the old state and the action.
        """
        try:
            return self.rewards[(self.curr_state, action)] * (
                self.beta**self.curr_time
            )
        except KeyError:
            # When accessing undefined states.
            return 0.0

    def sample(self):
        """
        Samples an action from the action space
        """
        return self.action_space.sample()


def create_mdp_and_run_epoch(args):
    """ 
    Function to create an MDP and run an epoch, to get the average reward.
    Parameters are hardcoded for now.
    """
    # simulating the text example.
    mdp_states, mdp_actions = 3, 2

    # rewards are represented as (curr_state,action) key-value paris
    mdp_rewards = {(0, 0): -5, (0, 1): 10, (1, 0): 5, (1, 1): 0, (2, 0): 20}

    # transition probabilities are represented as
    # (new_state, old_state, action): dict key-value pairs
    mdp_transitions = {
        (0, 0): [0.9, 0, 0.1],
        (0, 1): [0, 1, 0],
        (1, 0): [0, 1, 0],
        (1, 1): (0.8, 0.2, 0),
        (2, 0): [0.9, 0.1, 0],
    }

    # Strategy used by the MDP
    mdp_strategy = [(0.1, 0.9), (1., 0.), (1.,)]

    # just checking
    assert mdp_transitions.keys() == mdp_rewards.keys()

    mdp = MDP(
        mdp_actions,
        mdp_states,
        mdp_rewards,
        mdp_transitions,
        init_state=2,
        beta=0.8,
        f=mdp_strategy,
        timesteps=1000,
    )

    done = False
    while not done:
        curr_action = mdp.sample()
        curr_state, reward, done, info = mdp.step(curr_action)
    result = mdp.curr_reward
    mdp.reset()
    
    # return the result of the trajectory
    return result

if __name__ == "__main__":
    num_epochs, record_list = 100, []
    # running it in parallel for speed
    with mp.Pool() as pool: 
        for result in pool.map(create_mdp_and_run_epoch, range(num_epochs)):
            record_list.append(result)

    # checking that we did not lose any results
    assert (len(record_list) == num_epochs)
    
    average_reward = sum(record_list) / len(record_list)
    print(f"Average reward over {num_epochs} runs = {average_reward}")
    print(f"Max reward over {num_epochs} runs = {max(record_list)}")
    print(f"Min reward over {num_epochs} runs = {min(record_list)}")
