"""
Module to register the models created in this directory.
Also helps make them exportable to other directories.
"""
from gym.envs.registration import register

register(
    id='src/SummableMDP',
    entry_point='mdp:SummableMDP',
    max_episode_steps=300,
)
