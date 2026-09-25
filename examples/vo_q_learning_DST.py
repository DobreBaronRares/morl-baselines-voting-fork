import time

import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from morl_baselines.common.evaluation import eval_mo
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import MPMOQLearning
from morl_baselines.single_policy.ser.voq_learning import VoQLearning
from morl_baselines.multi_policy.multi_policy_voqlearning.mp_voq_learning import MPVoQLearning
from morl_baselines.common.voting_methods import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == "__main__":
    env = MORecordEpisodeStatistics(mo_gym.make("mo-reacher-v5"), gamma=0.99)
    eval_env = mo_gym.make("mo-reacher-v5")
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    ref_point = np.array([0., 0., 0., 0.])

    #voting_algos = [plurality_voting, rated_voting, topmost_median_rank, topmost_mean_rank, highest_median_voting, highest_mean_voting, two_round_voting, plurality_with_instant_runoff, STAR_voting, BTAR_voting, ATAR_voting, three_two_one_voting, maximin_voting, round_robin_voting, nanson_method, baldwin_method, rouse_method, cardinal_baldwin, instant_runoff_voting, coombs_method, carey_method, bucklin_voting, anti_bucklin_voting, oklahoma_primary_voting, fallback_voting, blacks_method, graduated_majority_judgement, cumulative_voting, quadratic_voting, vote_for_and_against, exhaustive_ballot, bottom_two_runoff_IRV, condorcet_fpp, condorcet_IRV, weighted_positional_methods, kotze_pereira_transform, keener_eigenvalues, prob_range_voting, small_method, first_preference_copeland, dodgson_method, simpson_method, reynaud_method, minimax_condorcet_method, mmpo, brbo, cross_max, contingent_vote, sri_lankan_contingent_vote, supplementary_vote, distributed_voting, kemeny_method, maxparc, maxparc_approval, maximal_lottery, fplossa, moral_parliament, proportional_borda, mdd_fpp, rcipe, koth, voice_of_reason, tacc, dsc, schulze_method, tidemans_alternative, woodall, plurality_veto, consensus_builder, benhams_method, ranked_pairs, river_method, stable_voting, sortition, random_ballot, random_pair, worst_winner]
    voting_algos = [range_voting, highest_mean_voting, cardinal_baldwin, instant_runoff_voting, graduated_majority_judgement, cumulative_voting, quadratic_voting, condorcet_fpp, condorcet_IRV, kotze_pereira_transform, prob_range_voting, first_preference_copeland, dodgson_method, kemeny_method, rcipe, dsc, river_method]
    #for voting_alg in voting_algos:
    agent = MPMOQLearning(env, log=True)
    agent.train(
        total_timesteps=int(1e7),
        ref_point=ref_point,
        eval_freq=1000,
        eval_env=eval_env,
    )

    print(eval_mo(agent, env=eval_env, w=weights))
