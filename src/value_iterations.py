"""
Module to implement the value iterations method for Markov Decision Processes.
This will determine the right strategy values.
"""

from mdp import *
import matplotlib.pyplot as plt


def value_iterations(mdp: MDP, eps: float = 1e-5) -> list[float]:
    """
    Function to implement the value iterations method for
    a Markov Decision Process.
    """
    V = np.zeros(mdp.observation_space.n)
    pi = np.zeros(mdp.observation_space.n)
    delta = 2 * eps
    while delta > eps:
        q = np.zeros((mdp.observation_space.n, mdp.action_space.n))
        for s in range(mdp.observation_space.n):
            v_curr = V[s]
            for a in range(mdp.action_space.n):
                try:
                    q[s, a] = mdp.rewards[s, a] + mdp.beta * sum(
                        [
                            mdp.transitions[(s, a)][s_prime] * V[s_prime]
                            for s_prime in range(mdp.observation_space.n)
                        ]
                    )
                except KeyError:
                    # Passing over undefined states
                    pass
            V[s] = np.max(q[s, :])
            pi[s] = np.argmax(q[s, :])
            delta = abs(V[s] - v_curr)

    # return calculated policy and values
    return V, pi


if __name__ == "__main__":
    # create the same MDP as before
    mdp_states, mdp_actions = 3, 2

    # rewards are represented as (curr_state,action)->reward key-value pairs
    mdp_rewards = {(0, 0): -5, (0, 1): 10, (1, 0): 5, (1, 1): 0, (2, 0): 20}

    # transition probabilities are represented as a dict of
    # (curr_state, action)-> proba_of_state[idx] key-value pairs
    mdp_transitions = {
        (0, 0): [0.9, 0, 0.1],
        (0, 1): [0, 1, 0],
        (1, 0): [0, 1, 0],
        (1, 1): (0.8, 0.2, 0),
        (2, 0): [0.9, 0.1, 0],
    }

    # Strategy used by the MDP
    # strat[state] = list[proba_of_choosing_action: float]
    # Note: here, state 3 only defines action 1, so it is
    #       a singleton.
    mdp_strategy = [(0.1, 0.9), (1.0, 0.0), (1.0,)]

    # just checking
    assert mdp_transitions.keys() == mdp_rewards.keys()

    mdp = MDP(
        mdp_actions,
        mdp_states,
        mdp_rewards,
        mdp_transitions,
        init_state=1,
        beta=0.8,
        f=mdp_strategy,
        timesteps=1000,
    )

    # Run the value iterations to get optimal values and policy
    print(f"Results of value_iterations: {value_iterations(mdp)}")
