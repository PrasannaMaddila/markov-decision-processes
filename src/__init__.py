from gym.envs.registration import register

register(
    id='src/SummableMDP',
    entry_point='mdp:SummableMDP',
    max_episode_steps=300,
)

