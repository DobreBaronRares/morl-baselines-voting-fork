import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from morl_baselines.common.scalarization import tchebicheff
from morl_baselines.multi_policy.multi_policy_voqlearning.mp_voq_learning import (
    MPVoQLearning,
)
from morl_baselines.common.voting_methods import *


if __name__ == "__main__":
    env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.99)
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0")
    scalarization = tchebicheff(tau=4.0, reward_dim=2)

    mp_moql = MPVoQLearning(
        env,
        learning_rate=0.3,
        voting_method=plurality_voting,
        dyna=False,
        initial_epsilon=1,
        final_epsilon=0.01,
        epsilon_decay_steps=int(2e5),
        weight_selection_algo="random",
        epsilon_ols=0.0,
    )
    mp_moql.train(
        total_timesteps=15 * int(2e5),
        timesteps_per_iteration=int(2e5),
        eval_freq=100,
        num_eval_episodes_for_front=1,
        eval_env=eval_env,
        ref_point=np.array([0.0, -25.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
    )
