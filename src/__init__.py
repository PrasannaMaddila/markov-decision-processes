"""
Module to register the models created in this directory.
Also helps make them exportable to other directories.
"""
from gymnasium.envs.registration import register

register(
    id="src/MDP",
    entry_point="mdp:MDP",
    max_episode_steps=100,
)
