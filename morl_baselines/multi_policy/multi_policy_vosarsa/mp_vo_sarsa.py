"""Outer-loop VoSARSA algorithm (uses multiple weights)."""

import time
from copy import deepcopy
from typing import List, Optional
from typing_extensions import override

import gymnasium as gym
import numpy as np

from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,
    policy_evaluation_mo,
)
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.voting_methods import plurality_voting
from morl_baselines.common.weights import equally_spaced_weights, random_weights
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from morl_baselines.single_policy.ser.vo_sarsa import VoSARSA


class MPVoSARSA(MOAgent):
    """Multi-policy VoSARSA: Outer loop version of vo_sarsa.

    Paper: Paper: K. Van Moffaert, M. Drugan, and A. Nowe, Scalarized Multi-Objective Reinforcement Learning: Novel Design Techniques. 2013. doi: 10.1109/ADPRL.2013.6615007.
    """

    def __init__(
        self,
        env,
        voting_method=plurality_voting,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        initial_epsilon: float = 0.1,
        final_epsilon: float = 0.1,
        epsilon_decay_steps: int = None,
        weight_selection_algo: str = "random",
        epsilon_ols: Optional[float] = None,
        transfer_q_table: bool = True,
        dyna: bool = False,
        dyna_updates: int = 5,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "MultiPolicy VoSARSA",
        wandb_entity: Optional[str] = None,
        seed: Optional[int] = None,
        log: bool = True,
    ):
        """Initialize the Multi-policy VoSARSA algorithm.

        Args:
            env: The environment to learn from.
            voting_method: The voting algorithm to use.
            learning_rate: The learning rate.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            final_epsilon: The final epsilon value.
            epsilon_decay_steps: The number of steps for epsilon decay.
            weight_selection_algo: The algorithm to use for weight selection. Options: "random", "ols"
            epsilon_ols: The epsilon value for the optimistic linear support.
            transfer_q_table: Whether to reuse a Q-table from a previous learned policy when initializing a new policy.
            dyna: Whether to use Dyna-Q or not.
            dyna_updates: The number of Dyna-Q updates to perform.
            project_name: The name of the project for logging.
            experiment_name: The name of the experiment for logging.
            wandb_entity: The entity to use for logging.
            seed: The seed to use for reproducibility.
            log: Whether to log or not.
        """
        MOAgent.__init__(self, env, seed=seed)
        # Learning
        self.voting_method = voting_method
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.dyna = dyna
        self.dyna_updates = dyna_updates
        self.transfer_q_table = transfer_q_table
        # Linear support
        self.policies = []
        self.weight_selection_algo = weight_selection_algo
        self.epsilon_ols = epsilon_ols
        assert self.weight_selection_algo in [
            "random",
            "ols",
        ], f"Unknown weight selection algorithm: {self.weight_selection_algo}."
        self.linear_support = LinearSupport(num_objectives=self.reward_dim, epsilon=epsilon_ols)

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        if self.log:
            self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name+"__"+voting_method.__name__, entity=wandb_entity)

    @override
    def get_config(self) -> dict:
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "voting_method": self.voting_method.__name__,
            "weight_selection_algo": self.weight_selection_algo,
            "epsilon_ols": self.epsilon_ols,
            "transfer_q_table": self.transfer_q_table,
            "dyna": self.dyna,
            "dyna_updates": self.dyna_updates,
            "seed": self.seed,
        }

    def eval(self, obs: np.array, w: Optional[np.ndarray] = None) -> int:
        """Chooses the best policy for w and follows it."""
        best_policy = np.argmax([np.dot(w, v) for v in self.linear_support.ccs])
        return self.policies[best_policy].eval(obs, w)

    def delete_policies(self, delete_indx: List[int]):
        """Delete the policies with the given indices."""
        for i in sorted(delete_indx, reverse=True):
            self.policies.pop(i)

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        timesteps_per_iteration: int = int(2e5),
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        eval_freq: int = 1000,
    ):
        """Learn a set of policies.

        Args:
            total_timesteps: The total number of timesteps to train for.
            eval_env: The environment to use for evaluation.
            ref_point: The reference point for the hypervolume calculation.
            known_pareto_front: The optimal Pareto front, if known. Used for metrics.
            timesteps_per_iteration: The number of timesteps per iteration.
            num_eval_weights_for_front: The number of weights to use to construct a Pareto front for evaluation.
            num_eval_episodes_for_front: The number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            eval_freq: The frequency of evaluation.
        """
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "timesteps_per_iteration": timesteps_per_iteration,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "eval_freq": eval_freq,
                }
            )
        num_iterations = int(total_timesteps / timesteps_per_iteration)
        if eval_env is None:
            eval_env = deepcopy(self.env)

        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)

        for iter in range(num_iterations):
            if self.weight_selection_algo == "ols":
                w = self.linear_support.next_weight(
                    algo=self.weight_selection_algo,
                    gpi_agent=None,
                    env=None,
                    rep_eval=num_eval_episodes_for_front,
                )
                if w is None:
                    print("OLS has no more corner weights to try. Using a random weight instead.")
                    w = random_weights(self.reward_dim, rng=self.np_random)
            elif self.weight_selection_algo == "random":
                w = random_weights(self.reward_dim, rng=self.np_random)

            if len(self.policies) == 0 or not self.dyna:
                model = None
            else:
                model = self.policies[-1].model  # shared model
            new_agent = VoSARSA(
                env=self.env,
                id=iter,
                weights=w,
                voting_method=self.voting_method,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                initial_epsilon=self.initial_epsilon,
                final_epsilon=self.final_epsilon,
                epsilon_decay_steps=self.epsilon_decay_steps,
                dyna=self.dyna,
                dyna_updates=self.dyna_updates,
                model=model,
                parent=self,
                log=self.log,
                parent_rng=self.np_random,
                seed=self.seed,
            )
            if self.transfer_q_table and len(self.policies) > 0:
                reuse_ind = np.argmax([np.dot(w, v) for v in self.linear_support.ccs])
                new_agent.q_table = deepcopy(self.policies[reuse_ind].q_table)
            self.policies.append(new_agent)

            start_time = time.time()
            new_agent.global_step = self.global_step
            new_agent.train(
                start_time=start_time,
                total_timesteps=timesteps_per_iteration,
                reset_num_timesteps=False,
                eval_freq=eval_freq,
                eval_env=eval_env,
            )
            self.global_step = new_agent.global_step

            value = policy_evaluation_mo(agent=new_agent, env=eval_env, w=w, rep=num_eval_episodes_for_front)[3]
            removed_inds = self.linear_support.add_solution(value, w)
            if self.weight_selection_algo != "random":
                self.delete_policies(removed_inds)

            if self.log:
                front = self.linear_support.ccs
                log_all_multi_policy_metrics(
                    current_front=front,
                    hv_ref_point=ref_point,
                    reward_dim=self.reward_dim,
                    global_step=self.global_step,
                    n_sample_weights=num_eval_weights_for_eval,
                    ref_front=known_pareto_front,
                )

        if self.log:
            self.close_wandb()
