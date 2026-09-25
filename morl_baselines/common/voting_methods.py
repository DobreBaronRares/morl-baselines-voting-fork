"""Voting methods for Extended Voting Q-learning."""

import numpy as np
import copy
from functools import wraps
from itertools import repeat
from pref_voting import margin_based_methods, iterative_methods
from pref_voting.profiles import Profile
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize, LinearConstraint, Bounds
from sympy.utilities.iterables import multiset_permutations
import wandb

"""
UTILS
"""
def pairwise_defeat_matrix(reward_action: np.ndarray):
    a = reward_action.shape[0]
    r = reward_action.shape[1]
    matrix = np.zeros((a, a), dtype=int)

    for reward in range(r):
        for candidate1 in range(a):
            for candidate2 in range(a):
                if (candidate1 != candidate2) and reward_action[candidate1, reward] > reward_action[candidate2, reward]:
                    matrix[candidate1][candidate2] += 1
                #elif (candidate1 != candidate2) and reward_action[candidate1, reward] == reward_action[candidate2, reward]:
                #    matrix[candidate1][candidate2] += 0.5

    return matrix

def pairwise_victory_score(reward_action: np.ndarray, victory_score: float, equal_score: float):
    half_action_shape = reward_action.shape[0] / 2.
    score = np.zeros(reward_action.shape[0])
    for action1 in range(reward_action.shape[0] - 1):
        for action2 in range(action1 + 1, reward_action.shape[0]):
            win_count = np.count_nonzero(np.greater(reward_action[action1], reward_action[action2]))
            if win_count == half_action_shape:
                score[action1] += equal_score
                score[action2] += equal_score
            elif win_count > half_action_shape:
                score[action1] += victory_score
            else:
                score[action2] += victory_score
    return score

def normalize(reward_action):
    scaler = MinMaxScaler()
    scaler.fit(reward_action)
    reward_action_scaled = scaler.transform(reward_action)
    return reward_action_scaled

def make_bins(reward_action: np.ndarray, grades: int):
    reward_action_norm = normalize(reward_action)
    bins = np.array(list(range(grades + 1)), dtype='float64')
    bins /= bins[-1]
    bins[0] -= 0.01
    reward_action_inds = np.apply_along_axis(np.digitize, axis=1, arr=reward_action_norm, bins=bins, right=True)
    reward_action_inds -= 1
    return reward_action_inds

# Voting function decorator for every function.
# Actions are considered candidates. Rewards are considered voters.
# Each ballot, for each reward, is generated from the list of individual reward values from each action.
def voting(voting_alg):
    @wraps(voting_alg)
    def wrapper_alg(reward_action: np.ndarray, weights: np.ndarray=None, *args, **kwargs) -> int:
        """
        Args:
            reward_action: Vector for action and reward tables (dimension: [n_actions, n_rewards])
            weights: Vector of reward weights

        Returns:
            int: Index of action voted to be the best by all rewards (or chosen randomly for ties)
        """
        if weights is None:
            weights = np.ones(reward_action.shape[1])
        # Indexes of the winners
        winners = voting_alg(np.squeeze(reward_action * np.squeeze(weights)), *args, **kwargs)
        if np.array_equal(winners, np.zeros(winners.shape)):
            winners = np.ones(winners.shape)
        # Indexes of all actions
        indexes = range(winners.shape[0])
        # Normalized probabilities of all winners
        probs = winners * np.ones(reward_action.shape[0])
        if np.sum(probs) == 0.:
            probs = np.ones(winners.shape)
        probs = probs / np.sum(probs)
        # Choose action from the winners using probabilities
        chosen_action = np.random.choice(indexes, p=probs)
        # Bayesian Regret (BR, https://www.rangevoting.org/BRworked.txt)
        br = np.maximum(0., np.max(np.sum((reward_action * np.squeeze(weights)), axis=1)) - np.sum((reward_action * np.squeeze(weights)), axis=1)[chosen_action])
        wandb.log(
            {
                f"bayesian_regret": br,
            },
        )
        return chosen_action.item()
    return wrapper_alg

"""
EXISTANT VOTING ALGORITHMS
"""

@voting
def approval_voting(reward_action: np.ndarray, threshold: str="mean") -> np.array:
    """
    Name: Approval voting
    Desc: If the reward value is above a threshold, the action receives a vote.
          Action with the largest number of votes wins.
    Source: second VoQL paper (https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1632&context=etd_projects)
    """
    if threshold == "mean":
        comps = np.mean(reward_action, axis=0)
    elif threshold == "median":
        comps = np.median(reward_action, axis=0)
    elif threshold == "t3":
        comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 3. * 2.
    elif threshold == "t1":
        comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 3.
    elif threshold == "q3":
        comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 4. * 3.
    elif threshold == "q1":
        comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 4.
    else:
        comps = np.mean(reward_action, axis=0)
    points = np.greater_equal(reward_action, comps)
    score = np.count_nonzero(points, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def range_voting(reward_action: np.ndarray, normalized: bool=True) -> np.array:
    """
    Name: Range voting, Optimum winner
    Desc: For each action, reward values are summed.
          Action with the highest sum is picked.
          If normalized: resembles Optimum winner (best possible winner!)
    Source: original VoQL paper (https://www.sciencedirect.com/science/article/abs/pii/S0957417416305863)
    """
    if normalized:
        reward_action_norm = normalize(reward_action)
    else:
        reward_action_norm = reward_action
    score = np.sum(reward_action_norm, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def borda_count(reward_action: np.ndarray) -> np.array:
    """
    Name: Borda count
    Desc: Each reward gives an ordering of its preferences.
          The indices of all preferences are summed for each action (best of n actions has index n-1, last has 0).
    Source: original VoQL paper (https://www.sciencedirect.com/science/article/abs/pii/S0957417416305863)
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    score = np.sum(order, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def copeland_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Copeland voting, Llull method
    Desc: For each action, pairwise elections are held with all other actions.
          Each action's score is the total of its wins, plus half the total of its draws.
          Action with largest numer of points wins.
    Source: original VoQL paper (https://www.sciencedirect.com/science/article/abs/pii/S0957417416305863)
    """
    score = pairwise_victory_score(reward_action, 1.0, 0.5)
    winners = score == np.max(score)
    return winners

@voting
def negative_voting(reward_action: np.ndarray, approval_schema: str="thirds") -> np.array:
    """
    Name: Negative voting, Combined approval voting
    Desc: If the reward value is above an approval threshold, the action receives a point.
          If the reward value is below an approval threshold, the action subtracts a point.
          Action with the largest number of points wins.
    Source: second VoQL paper (https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1632&context=etd_projects)
    """
    if approval_schema == "thirds":
        comps_app = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 3. * 2.
        comps_dpp = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 3.
    elif approval_schema == "quarters":
        comps_app = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 4. * 3.
        comps_dpp = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 4.
    else:
        comps_app = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 3. * 2.
        comps_dpp = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 3.
    points_app = np.greater_equal(reward_action, comps_app)
    points_dpp = np.less_equal(reward_action, comps_dpp)
    score = np.count_nonzero(points_app, axis=1) - np.count_nonzero(points_dpp, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def plurality_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Plurality voting, First-past-the-post (FPTP)
    Desc: Action with the largest reward value receives a vote.
          Action with the most votes wins.
    Source: original VoQL paper (https://www.sciencedirect.com/science/article/abs/pii/S0957417416305863), as "Approval voting"
    """
    max_actions = np.argmax(reward_action, axis=0)
    votes = np.bincount(max_actions, minlength=reward_action.shape[0])
    winners = votes == np.max(votes)
    return winners

"""
ADDITIONAL VOTING ALGORITHMS
"""

@voting
def anti_plurality_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Anti-plurality voting
    Desc: Action with the lowest reward value receives a negative vote.
          Action with the least negative votes wins.
    Source: original VoQL paper (https://www.sciencedirect.com/science/article/abs/pii/S0957417416305863), as "Approval voting"
    """
    min_actions = np.argmin(reward_action, axis=0)
    votes = np.bincount(min_actions, minlength=reward_action.shape[0])
    winners = votes == np.min(votes)
    return winners

@voting
def rated_voting(reward_action: np.ndarray, grades: int=5) -> np.array:
    """
    Name: Rated voting, Graded voting, Cardinal voting, Evaluative voting, Score voting
    Desc: Given a number of grades, each reward gives a grade to each action based on the bin of its value.
          Action with largest sum of grades wins.
    Source: N/A
    """    
    reward_action_inds = make_bins(reward_action, grades)
    score = np.sum(reward_action_inds, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def topmost_median_rank(reward_action: np.ndarray) -> np.array:
    """
    Name: Topmost Median Rank 
    Desc: Each reward gives an ordering of its preferences.
          Action chosen has the maximum median index.
    Source: http://www.9mail.de/m-schulze/votedesc.pdf
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    score = np.median(order, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def topmost_mean_rank(reward_action: np.ndarray) -> np.array:
    """
    Name: Topmost Mean Rank 
    Desc: Each reward gives an ordering of its preferences.
          Action chosen has the maximum mean index.
    Source: N/A
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    score = np.round(np.mean(order, axis=1))
    winners = score == np.max(score)
    return winners

@voting
def highest_median_voting(reward_action: np.ndarray, range: bool=True, grades: int=5) -> np.array:
    """
    Name: Highest median voter
    Desc: Given a number of grades, each reward gives a grade to each action based on the bin of its value OR its value.
          Action with largest median of grades wins.
    Source: N/A
    """
    reward_action_norm = normalize(reward_action)

    if not range:
        reward_action_norm = make_bins(reward_action_norm, grades)

    score = np.median(reward_action_norm, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def highest_mean_voting(reward_action: np.ndarray, range: bool=True, grades: int=5) -> np.array:
    """
    Name: Highest mean voter
    Desc: Given a number of grades, each reward gives a grade to each action based on the bin of its value OR its value.
          Action with largest mean of grades wins.
    Source: N/A
    """
    reward_action_norm = normalize(reward_action)

    if not range:    
        reward_action_norm = make_bins(reward_action_norm, grades)

    score = np.mean(reward_action_norm, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def two_round_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Two-round voting
    Desc: Action with the largest reward value receives a vote.
          Process is repeated for the top two actions.
          Action with the most votes wins.
    Source: N/A
    """
    # Round 1
    max_actions = np.argmax(reward_action, axis=0)
    votes = np.bincount(max_actions, minlength=reward_action.shape[0])
    winners = (votes == np.max(votes)) | (votes == np.sort(votes)[-2])

    # Round 2
    top_reward_action = -np.inf * np.ones_like(reward_action)
    top_reward_action[winners] = reward_action[winners]
    top_max_actions = np.argmax(reward_action, axis=0)
    top_votes = np.bincount(top_max_actions, minlength=reward_action.shape[0])
    top_winners = (top_votes == np.max(top_votes))
    return top_winners

@voting
def plurality_with_instant_runoff(reward_action: np.ndarray) -> np.array:
    """
    Name: Plurality with instant runoff
    Desc: Action with the largest reward value receives a vote.
          Top two actions are compared directly.
    Source: http://www.9mail.de/m-schulze/votedesc.pdf
    """
    # Round 1
    max_actions = np.argmax(reward_action, axis=0)
    votes = np.bincount(max_actions, minlength=reward_action.shape[0])
    winners = (votes == np.max(votes)) | (votes == np.sort(votes)[-2])

    # Round 2
    top_reward_action = reward_action[winners]
    half_action_shape = reward_action.shape[0] / 2.
    win_count = np.count_nonzero(np.greater(top_reward_action[0], top_reward_action[1]))
    if win_count > half_action_shape:
        top_winners = (reward_action == top_reward_action[0]).all(axis=1)
    else:
        top_winners = (reward_action == top_reward_action[1]).all(axis=1)
    return top_winners

@voting
def STAR_voting(reward_action: np.ndarray, grades: int=5) -> np.array:
    """
    Name: STAR voting, Score Then Automatic Runoff
    Desc: Given a number of grades, each reward gives a grade to each action based on the bin of its value.
          Two actions with largest sum of grades are selected.
          Action with larger score on the most ballots wins.
    Source: N/A
    """
    # Score    
    reward_action_inds = make_bins(reward_action, grades)

    score = np.sum(reward_action_inds, axis=1)
    winners = (score == np.max(score)) | (score == np.sort(score)[-2])

    # Count
    top_reward_action = -np.inf * np.ones_like(reward_action)
    top_reward_action[winners] = reward_action[winners]
    top_max_actions = np.argmax(top_reward_action, axis=0)
    top_votes = np.bincount(top_max_actions, minlength=reward_action.shape[0])
    top_winners = (top_votes == np.max(top_votes))
    return top_winners

@voting
def BTAR_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: BTAR voting, Borda Then Automatic Runoff
    Desc: Each reward gives an ordering of its preferences.
          The indices of all preferences are summed for each action (best of n actions has index n-1, last has 0).
          Two actions with largest sum of grades are selected.
          Action with larger score on the most ballots wins.
    Source: https://votingmethods.net/trunc/
    """
    # Borda
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    score = np.sum(order, axis=1)
    winners = (score == np.max(score)) | (score == np.sort(score)[-2])

    # Count
    top_reward_action = -np.inf * np.ones_like(reward_action)
    top_reward_action[winners] = reward_action[winners]
    top_max_actions = np.argmax(top_reward_action, axis=0)
    top_votes = np.bincount(top_max_actions, minlength=reward_action.shape[0])
    top_winners = (top_votes == np.max(top_votes))
    return top_winners

@voting
def ATAR_voting(reward_action: np.ndarray, threshold: str="mean") -> np.array:
    """
    Name: ATAR voting, Approval Then Automatic Runoff
    Desc: If the reward value is above a threshold, the action receives a vote.
          Two actions with largest amount of votes are selected.
          Action with larger score on the most ballots wins.
    Source: https://votingmethods.net/trunc/
    """
    # Approval
    if threshold == "mean":
        comps = np.mean(reward_action, axis=0)
    elif threshold == "median":
        comps = np.median(reward_action, axis=0)
    elif threshold == "t3":
        comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 3. * 2.
    elif threshold == "t1":
        comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 3.
    elif threshold == "q3":
        comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 4. * 3.
    elif threshold == "q1":
        comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 4.
    else:
        comps = np.mean(reward_action, axis=0)
    points = np.greater_equal(reward_action, comps)
    score = np.count_nonzero(points, axis=1)
    winners = (score == np.max(score)) | (score == np.sort(score)[-2])

    # Count
    top_reward_action = -np.inf * np.ones_like(reward_action)
    top_reward_action[winners] = reward_action[winners]
    top_max_actions = np.argmax(top_reward_action, axis=0)
    top_votes = np.bincount(top_max_actions, minlength=reward_action.shape[0])
    top_winners = (top_votes == np.max(top_votes))
    return top_winners

@voting
def three_two_one_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: 3-2-1 voting
    Desc: Actions are rated "Good", "Ok", "Bad".
          3 semifinalists: actions with the most "good" ratings.
          2 finalists: semifinalists with the fewest "bad" ratings.
          1 winner: the finalist who is rated above the other on the most ballots.
    Source: https://electowiki.org/wiki/3-2-1_voting
    """    
    reward_action_inds = make_bins(reward_action, 3)
    reward_action_inds += 1

    # 3 semifinalists
    count_3s = np.count_nonzero(reward_action_inds == 3, axis=1)
    counts_3s_max = np.sort(count_3s)[-3:]
    winner_3_indexes = np.isin(count_3s, counts_3s_max)
    reward_action_inds_3 = reward_action_inds[winner_3_indexes]
    # 2 finalists
    count_1s = np.count_nonzero(reward_action_inds_3 == 1, axis=1)
    counts_1s_max = np.sort(count_1s)[:2]
    winner_1_indexes = np.isin(count_1s, counts_1s_max)
    reward_action_inds_2 = reward_action_inds_3[winner_1_indexes]
    # 1 winner
    half_action_shape = reward_action_inds.shape[0] / 2.
    win_count = np.count_nonzero(np.greater(reward_action_inds_2[0], reward_action_inds_2[1]))
    if win_count > half_action_shape:
        winners = (reward_action_inds == reward_action_inds_2[0]).all(axis=1)
    else:
        winners = (reward_action_inds == reward_action_inds_2[1]).all(axis=1)
    return winners

@voting
def maximin_voting(reward_action: np.ndarray, normalized: bool=False) -> np.array:
    """
    Name: Maximin voting
    Desc: Action with the largest minimum reward value receives a vote.
          Action with the most votes wins.
    Source: N/A
    """
    if normalized:
        reward_action_norm = normalize(reward_action)
    else:
        reward_action_norm = reward_action
    min_values = np.min(reward_action_norm, axis=1)
    winners = min_values == np.max(min_values)
    return winners

@voting
def round_robin_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Round-robin voting, Paired comparison, Tournament voting
    Desc: For each action, pairwise elections are held with all other actions.
          Each action's score is the total of its wins.
          Action with largest numer of points wins.
    Source: https://en.wikipedia.org/wiki/Round-robin_voting
    """
    half_action_shape = reward_action.shape[0] / 2.
    score = np.zeros(reward_action.shape[0])
    for action1 in range(reward_action.shape[0] - 1):
        for action2 in range(action1 + 1, reward_action.shape[0]):
            win_count = np.count_nonzero(np.greater(reward_action[action1], reward_action[action2]))
            if win_count == half_action_shape:
                coin_toss = np.random.randint(0, 2, size=1)
                if coin_toss == 0:
                    score[action1] += 1.0
                else:
                    score[action2] += 1.0
            elif win_count > half_action_shape:
                score[action1] += 1.0
            else:
                score[action2] += 1.0
    winners = score == np.max(score)
    return winners

@voting
def nanson_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Nanson's method
    Desc: Do successive Borda counts, dropping the actions with values below the mean each time.
    Source: https://en.wikipedia.org/wiki/Nanson%27s_method
    """
    red_reward_action = copy.deepcopy(reward_action)
    score = np.array([0., 1.])
    while np.unique(score).shape[0] > 1:
        order = np.argsort(np.argsort(red_reward_action, axis=0), axis=0)
        score = np.sum(order, axis=1)
        min_score_inds = score < np.mean(score)

        red_reward_action = np.delete(red_reward_action, min_score_inds, axis=0)
        score = np.delete(score, min_score_inds, axis=0)
    winners = np.zeros(reward_action.shape[0], dtype=bool)
    for i in range(red_reward_action.shape[0]):
        winners = winners | (reward_action == red_reward_action[i]).all(axis=1)
    return winners

@voting
def baldwin_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Baldwin method
    Desc: Do successive Borda counts, dropping the action with the lowest value each time.
    Source: https://en.wikipedia.org/wiki/Nanson%27s_method#Baldwin_method
    """
    red_reward_action = copy.deepcopy(reward_action)
    score = np.array([0., 1.])
    while np.unique(score).shape[0] > 1:
        order = np.argsort(np.argsort(red_reward_action, axis=0), axis=0)
        score = np.sum(order, axis=1)
        min_score_inds = np.argmin(score)

        red_reward_action = np.delete(red_reward_action, min_score_inds, axis=0)
        score = np.delete(score, min_score_inds, axis=0)
    winners = np.zeros(reward_action.shape[0], dtype=bool)
    for i in range(red_reward_action.shape[0]):
        winners = winners | (reward_action == red_reward_action[i]).all(axis=1)
    return winners

@voting
def rouse_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Rouse method
    Desc: Do successive Borda counts, dropping the action given by successively eliminating actions with highest Borda count.
    Source: https://accuratedemocracy.com/download/software/rbvote/desc.html
    """
    red_reward_action = copy.deepcopy(reward_action)
    score = np.array([0., 1.])
    while np.unique(score).shape[0] > 1:
        order = np.argsort(np.argsort(red_reward_action, axis=0), axis=0)
        score = np.sum(order, axis=1)

        c_reward_action = copy.deepcopy(reward_action)
        c_order = copy.deepcopy(order)
        c_score = copy.deepcopy(score)
        while np.unique(c_score).shape[0] > 1:
            c_order = np.argsort(np.argsort(c_reward_action, axis=0), axis=0)
            c_score = np.sum(c_order, axis=1)
            max_score_ind = np.argmax(c_score, axis=0)
            c_reward_action = np.delete(c_reward_action, max_score_ind, axis=0)
            c_score = np.delete(c_score, max_score_ind, axis=0)
        
        min_score_ind = np.zeros_like(c_reward_action.shape[0])
        for i in range(c_reward_action.shape[0]):
            min_score_ind = min_score_ind and (red_reward_action == c_reward_action[i]).all(axis=1)
        red_reward_action = np.delete(red_reward_action, min_score_ind, axis=0)
        score = np.delete(score, min_score_ind, axis=0)
    
    winners = np.zeros(reward_action.shape[0], dtype=bool)
    for i in range(red_reward_action.shape[0]):
        winners = winners | (reward_action == red_reward_action[i]).all(axis=1)
    return winners

@voting
def cardinal_baldwin(reward_action: np.ndarray) -> np.array:
    """
    Name: Cardinal Baldwin
    Desc: Do successive score counts, dropping the action with the lowest value each time.
    Source: https://electowiki.org/wiki/Baldwin%27s_method#Cardinal_variant
    """
    reward_action_norm = normalize(reward_action)
    red_reward_action = copy.deepcopy(reward_action_norm)
    non_norm_reward_action = copy.deepcopy(reward_action_norm)
    init_score = np.sum(reward_action_norm, axis=1)
    score = np.array([0., 1.])
    while np.unique(score).shape[0] > 1:
        score = np.sum(red_reward_action, axis=1)
        min_score_ind = np.argmin(score)

        red_reward_action = np.delete(red_reward_action, min_score_ind, axis=0)
        non_norm_reward_action = np.delete(non_norm_reward_action, min_score_ind, axis=0)
        red_reward_action = normalize(red_reward_action)
        score = np.delete(score, min_score_ind, axis=0)
    init_score = np.sum(reward_action_norm, axis=1)
    f_score = np.sum(non_norm_reward_action, axis=1)
    winners = np.zeros(reward_action_norm.shape[0], dtype=bool)
    for i in range(f_score.shape[0]):
        winners = winners | (init_score == f_score[i])
    return winners

@voting
def instant_runoff_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Instant-Runoff Voting, IRV, Hare voting, Alternative voting
    Desc: Sequentially eliminates candidates with lowest top preferences.
    Source: https://en.wikipedia.org/wiki/Instant-runoff_voting
    """
    reward_action_inds = np.argsort(np.argsort(reward_action, axis=0), axis=0)

    grades = reward_action_inds.shape[0] - 1
    winners = np.ones(reward_action_inds.shape[0], dtype='bool')
    curr_votes = np.count_nonzero(reward_action_inds == grades, axis=1)
    while np.unique(curr_votes[winners]).shape[0] > 1:
        lowest_action = np.argmin(np.ma.MaskedArray(curr_votes, mask = ~winners))
        winners[lowest_action] = False
        reward_action_inds[~winners] = -1.
        reward_action_inds[winners] = np.argsort(np.argsort(reward_action[winners], axis=0), axis=0)
        grades -= 1
        curr_votes = np.count_nonzero(reward_action_inds == grades, axis=1)
    return winners

@voting
def coombs_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Coombs' method
    Desc: Sequentially eliminates candidates with most bottom preferences.
    Source: https://en.wikipedia.org/wiki/Coombs%27_method
    """
    reward_action_inds = np.argsort(np.argsort(reward_action, axis=0), axis=0)

    winners = np.ones(reward_action_inds.shape[0], dtype='bool')
    curr_votes = np.count_nonzero(reward_action_inds == 0, axis=1)
    while np.unique(curr_votes[winners]).shape[0] > 1:
        lowest_action = np.argmax(np.ma.MaskedArray(curr_votes, mask = ~winners))
        winners[lowest_action] = False
        reward_action_inds[~winners] = -1.
        reward_action_inds[winners] = np.argsort(np.argsort(reward_action[winners], axis=0), axis=0)
        curr_votes = np.count_nonzero(reward_action_inds == 0, axis=1)
    return winners

@voting
def carey_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Carey method
    Desc: Sequentially eliminates candidates with below mean top preferences.
    Source: https://accuratedemocracy.com/download/software/rbvote/desc.html
    """
    reward_action_inds = np.argsort(np.argsort(reward_action, axis=0), axis=0)

    grades = reward_action_inds.shape[0] - 1
    winners = np.ones(reward_action_inds.shape[0], dtype='bool')
    curr_votes = np.count_nonzero(reward_action_inds == grades, axis=1)
    while np.unique(curr_votes[winners]).shape[0] > 1:
        lowest_action = np.ma.MaskedArray(curr_votes, mask = ~winners) < np.mean(np.ma.MaskedArray(curr_votes, mask = ~winners))
        winners[lowest_action] = False
        reward_action_inds[~winners] = -1.
        reward_action_inds[winners] = np.argsort(np.argsort(reward_action[winners], axis=0), axis=0)
        grades -= 1
        curr_votes = np.count_nonzero(reward_action_inds == grades, axis=1)
    return winners

@voting
def bucklin_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Bucklin voting, Bucklin's rule
    Desc: Successively orders candidates by number of proponents until majority is achieved.
    Source: https://en.wikipedia.org/wiki/Bucklin_voting
    """
    reward_action_inds = np.argsort(np.argsort(reward_action, axis=0), axis=0)

    half_count = reward_action.shape[1] / 2.
    curr_ind = reward_action.shape[0] - 1
    curr_rank = (reward_action_inds == curr_ind)
    while np.max(np.count_nonzero(curr_rank, axis=1)) <= half_count:
        curr_ind -= 1
        curr_rank = np.logical_or(curr_rank, (reward_action_inds == curr_ind))

    winners = np.count_nonzero(curr_rank, axis=1) > half_count
    return winners

@voting
def anti_bucklin_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Anti-Bucklin voting, Anti-Bucklin's rule
    Desc: Successively orders candidates by number of opponents until majority is achieved.
    Source: https://en.wikipedia.org/wiki/Highest_median_voting_rules
    """
    reward_action_inds = np.argsort(np.argsort(reward_action, axis=0), axis=0)

    half_count = reward_action.shape[1] / 2.
    curr_ind = 0
    curr_rank = (reward_action_inds == curr_ind)
    while np.max(np.count_nonzero(curr_rank, axis=1)) <= half_count:
        curr_ind += 1
        curr_rank = np.logical_or(curr_rank, (reward_action_inds == curr_ind))

    winners = np.count_nonzero(curr_rank, axis=1) < half_count
    return winners

@voting
def oklahoma_primary_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Oklahoma primary electoral system, Bucklin-Dowdall hybrid
    Desc: Successively orders candidates by number of proponents until majority is achieved, with diminishing points.
    Source: https://en.wikipedia.org/wiki/Oklahoma_primary_electoral_system
    """
    reward_action_inds = np.argsort(np.argsort(reward_action, axis=0), axis=0)

    half_count = reward_action.shape[1] / 2.
    curr_ind = reward_action.shape[0] - 1
    curr_rank = (reward_action_inds == curr_ind)
    while np.max(np.sum(curr_rank, axis=1)) <= half_count:
        curr_ind -= 1
        curr_rank = curr_rank + (1. / (curr_ind + 1.)) * (reward_action_inds == curr_ind)

    winners = np.sum(curr_rank, axis=1) > half_count
    return winners

@voting
def fallback_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Fallback voting
    Desc: Successively orders candidates by number of proponents and approvals until majority is achieved.
    Source: https://electowiki.org/wiki/Bucklin_voting#Fallback_voting
    """
    reward_action_inds = np.argsort(np.argsort(reward_action, axis=0), axis=0)

    half_count = reward_action.shape[1] / 2.
    curr_ind = reward_action.shape[0] - 1
    curr_rank = (reward_action_inds == curr_ind)
    while np.max(np.count_nonzero(curr_rank, axis=1)) <= half_count and curr_ind >= reward_action.shape[0] / 2.:
        curr_ind -= 1
        curr_rank = np.logical_or(curr_rank, (reward_action_inds == curr_ind))

    if curr_ind < reward_action.shape[0] / 2.:
        winners = np.count_nonzero(curr_rank, axis=1) == np.max(np.count_nonzero(curr_rank, axis=1))
    else:
        winners = np.count_nonzero(curr_rank, axis=1) > half_count
    return winners

@voting
def blacks_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Black's method
    Desc: Choose Condorcet winner (Copeland). If none exists, choose Borda winner.
    Source: https://electowiki.org/wiki/Black%27s_method
    """
    # Copeland
    score = pairwise_victory_score(reward_action, 1.0, 0.5)
    winners = score == np.max(score)
    
    # Borda
    if np.count_nonzero(winners) > 1:
        order = np.argsort(np.argsort(reward_action, axis=0), axis=0)
        score = np.sum(order, axis=1)
        winners = score == np.max(score)

    return winners

@voting
def graduated_majority_judgement(reward_action: np.ndarray, range: bool=False, grades: int=5) -> np.array:
    """
    Name: Graduated Majority Judgment, GMJ, Usual judgment, Continuous Bucklin voting
    Desc: Continuous highest median voting.
    Source: https://en.wikipedia.org/wiki/Graduated_majority_judgment
    """
    reward_action_norm = normalize(reward_action)

    if not range:    
        reward_action_norm = make_bins(reward_action_norm, grades)

    median_score = np.median(reward_action_norm, axis=1)
    supporters_score = np.count_nonzero((reward_action_norm.T > median_score).T, axis=1) / reward_action_norm.shape[0]
    opponents_score = np.count_nonzero((reward_action_norm.T < median_score).T, axis=1) / reward_action_norm.shape[0]
    score = median_score + 0.5 * (supporters_score - opponents_score) / (1.0 - supporters_score - opponents_score)

    winners = score == np.max(score)
    return winners

@voting
def cumulative_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Cumulative voting
    Desc: Votes can be distributed among the actions.
    Source: https://en.wikipedia.org/wiki/Cumulative_voting
    """
    reward_action_norm = normalize(reward_action)
    votes = np.zeros(reward_action_norm.shape[0])
    for ballot in range(reward_action_norm.shape[1]):
        if np.max(reward_action_norm[:, ballot]) >= 3 * np.sort(reward_action_norm[:, ballot])[-2]:
            votes[np.argmax(reward_action_norm[:, ballot])] += 3
        elif np.max(reward_action_norm[:, ballot]) >= 2 * np.sort(reward_action_norm[:, ballot])[-2]:
            votes[np.argmax(reward_action_norm[:, ballot])] += 2
            votes[np.argsort(reward_action_norm[:, ballot])[-2]] += 1
        else:
            votes[np.argmax(reward_action_norm[:, ballot])] += 1
            votes[np.argsort(reward_action_norm[:, ballot])[-2]] += 1
            if reward_action_norm.shape[0] >= 3:
                votes[np.argsort(reward_action_norm[:, ballot])[-3]] += 1
    winners = (votes == np.max(votes))
    return winners

@voting
def quadratic_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Quadratic voting
    Desc: Votes can be distributed among the actions, with quadratic cost.
    Source: https://en.wikipedia.org/wiki/Quadratic_voting
    """
    reward_action_norm = normalize(reward_action)
    votes = np.zeros(reward_action_norm.shape[0])
    for ballot in range(reward_action_norm.shape[1]):
        if np.max(reward_action_norm[:, ballot]) >= 3 * np.sort(reward_action_norm[:, ballot])[-2]:
            votes[np.argmax(reward_action_norm[:, ballot])] += 9
            if reward_action_norm.shape[0] >= 3:
                votes[np.argsort(reward_action_norm[:, ballot])[-3]] += 1
        else:
            votes[np.argsort(reward_action_norm[:, ballot])[-2]] += 4
            if reward_action_norm.shape[0] >= 3:
                votes[np.argsort(reward_action_norm[:, ballot])[-3]] += 4
            if reward_action_norm.shape[0] >= 4:
                votes[np.argsort(reward_action_norm[:, ballot])[-4]] += 1
            if reward_action_norm.shape[0] >= 5:
                votes[np.argsort(reward_action_norm[:, ballot])[-5]] += 1
    winners = (votes == np.max(votes))
    return winners

@voting
def vote_for_and_against(reward_action: np.ndarray) -> np.array:
    """
    Name: Vote For and Against, VFA, Venzke Disqualified (or Disqualification) Plurality, VDP
    Desc: Action with the largest reward value receives a for vote.
          Action with the lowest reward value receives an against vote.
          Action with most for votes is picked, unless it has over half negative votes.
    Source: https://electowiki.org/wiki/Vote_For_and_Against
    """
    max_actions = np.argmax(reward_action, axis=0)
    min_actions = np.argmin(reward_action, axis=0)
    for_votes = np.bincount(max_actions, minlength=reward_action.shape[0])
    against_votes = np.bincount(min_actions, minlength=reward_action.shape[0])
    winners = (for_votes == np.max(for_votes)) & (against_votes < reward_action.shape[0] / 2.)
    return winners

@voting
def exhaustive_ballot(reward_action: np.ndarray) -> np.array:
    """
    Name: Exhastive ballot
    Desc: Action with the largest reward value receives a vote.
          Action with the most votes wins.
          If no candidate has a majority of votes, redo the procedure.
    Source: https://en.wikipedia.org/wiki/Exhaustive_ballot
    """
    red_reward_action = copy.deepcopy(reward_action)
    max_actions = np.argmax(reward_action, axis=0)
    votes = np.bincount(max_actions, minlength=reward_action.shape[0])
    while (np.max(votes) <= red_reward_action.shape[1] / 2.) and (red_reward_action != -np.inf * np.ones_like(red_reward_action)).all():
        nwinners = votes == np.min(votes)
        red_reward_action[nwinners] = -np.inf * np.ones(red_reward_action.shape[1])
        max_actions = np.argmax(red_reward_action, axis=0)
        votes = np.bincount(max_actions, minlength=red_reward_action.shape[0])
    winners = votes == np.max(votes)
    return winners

@voting
def bottom_two_runoff_IRV(reward_action: np.ndarray) -> np.array:
    """
    Name: Bottom-two-runoff IRV, BTR-IRV
    Desc: Sequentially the two options with the lowest top preferences are compared.
          The pairwise loser is eliminated.
    Source: https://electowiki.org/wiki/Bottom-Two-Runoff_IRV
    """
    reward_action_inds = np.argsort(np.argsort(reward_action, axis=0), axis=0)

    grades = reward_action_inds.shape[0] - 1
    winners = np.ones(reward_action_inds.shape[0], dtype='bool')
    curr_votes = np.count_nonzero(reward_action_inds == grades, axis=1)
    while np.unique(curr_votes[winners]).shape[0] > 1:
        first_lowest_action = np.argmin(np.ma.MaskedArray(curr_votes, mask = ~winners))
        mask_2 = ~winners[:]
        mask_2[first_lowest_action] = True
        second_lowest_action = np.argmin(np.ma.MaskedArray(curr_votes, mask = mask_2))
        if np.count_nonzero(reward_action_inds[first_lowest_action] < reward_action_inds[second_lowest_action]) < reward_action_inds.shape[1] / 2.0:
            lowest_action = second_lowest_action
        else:
            lowest_action = first_lowest_action

        winners[lowest_action] = False
        reward_action_inds[~winners] = -1.
        reward_action_inds[winners] = np.argsort(np.argsort(reward_action[winners], axis=0), axis=0)
        grades -= 1
        curr_votes = np.count_nonzero(reward_action_inds == grades, axis=1)

    return winners

@voting
def condorcet_fpp(reward_action: np.ndarray) -> np.array:
    """
    Name: Condorcet FPP
    Desc: Give Condorcet (Copeland) winner if it exists.
          Otherwise, give FPTP winner.
    Source: https://votingmethods.net/iiastrat/
    """
    # Condorcet (Copeland)
    score = pairwise_victory_score(reward_action, 1.0, 0.5)
    winners = score == np.max(score)

    # FPP
    if np.count_nonzero(winners) > 1:
        max_actions = np.argmax(reward_action, axis=0)
        votes = np.bincount(max_actions, minlength=reward_action.shape[0])
        winners = votes == np.max(votes)

    return winners

@voting
def condorcet_IRV(reward_action: np.ndarray) -> np.array:
    """
    Name: Condorcet IRV
    Desc: Give Condorcet (Copeland) winner if it exists.
          Otherwise, give IRV winner.
    Source: https://electowiki.org/wiki/Condorcet_IRV
    """
    # Condorcet (Copeland)
    score = pairwise_victory_score(reward_action, 1.0, 0.5)
    winners = score == np.max(score)

    # IRV
    if np.count_nonzero(winners) > 1:
        reward_action_inds = np.argsort(np.argsort(reward_action, axis=0), axis=0)

        grades = reward_action_inds.shape[0] - 1
        winners = np.ones(reward_action_inds.shape[0], dtype='bool')
        curr_votes = np.count_nonzero(reward_action_inds == grades, axis=1)
        while np.unique(curr_votes).shape[0] > 1:
            lowest_action = np.argmin(curr_votes)
            winners[lowest_action] = False
            reward_action_inds[~winners] = -1.
            reward_action_inds[winners] = np.argsort(np.argsort(reward_action[winners], axis=0), axis=0)
            grades -= 1
            curr_votes = np.count_nonzero(reward_action_inds == grades, axis=1)

    return winners

@voting
def weighted_positional_methods(reward_action: np.ndarray, weight_type: str='other') -> np.array:
    """
    Name: Weighted positional methods
    Desc: Each reward gives an ordering of its preferences.
          The indices of all preferences are given values based on the weight type.
          The values are summed for each action.
    Source: https://electowiki.org/wiki/Weighted_positional_method
    """
    weights = np.zeros(reward_action.shape[0])
    if weight_type == 'Nauru' or weight_type == 'Dowdall':
        weights = np.ones(reward_action.shape[0]) / (np.array(range(reward_action.shape[0])) + 1.0)[::-1]
    elif weight_type == 'Eurovision':
        weights[-1] = 12
        weights[-2] = 10
        weights[np.max([0, reward_action.shape[0]-2-8]):-2] = np.array(range(8))[-np.min([8, reward_action.shape[0]-2]):] + 1.0
    elif weight_type == 'BaseballMVP':
        weights[-1] = 14
        weights[np.max([0, reward_action.shape[0]-1-9]):-1] = np.array(range(9))[-np.min([9, reward_action.shape[0]-1]):] + 1.0
    elif weight_type == 'Heisman':
        weights[np.max([0, reward_action.shape[0]-3]):] = np.array(range(3))[-np.min([3, reward_action.shape[0]]):] + 1.0
    elif weight_type == 'Opt':
        weights[-1] = 1
        weights[-2] = 0.3723
    elif weight_type == 'Dabagh' or weight_type == 'Vote-and-a-half':
        weights[-1] = 1
        weights[-2] = 0.5
    elif weight_type == 'Binary' or weight_type == 'Geometric':
        range_array = np.array(range(reward_action.shape[0]))
        weights = np.power(0.5, np.flip(range_array))
    elif weight_type == 'D21':
        weights[np.max([0, reward_action.shape[0]-3]):] = np.ones(3)[-np.min([3, reward_action.shape[0]]):]
        weights[0] = -1
    else:
        weights = (np.array(range(reward_action.shape[0])) + 1.0) / reward_action.shape[0]
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    score = np.sum(weights[order], axis=1)
    winners = score == np.max(score)
    return winners

@voting
def kotze_pereira_transform(reward_action: np.ndarray, grades: int=5) -> np.array:
    """
    Name: Kotze-Pereira (KP) transform
    Desc: Converts score vote ballots to approval ballots
    Source: https://electowiki.org/wiki/Kotze-Pereira_transformation
    """    
    reward_action_inds = make_bins(reward_action, grades)

    groups = np.zeros_like(reward_action)
    for threshold in range(grades):
        groups += np.where(reward_action_inds >= threshold, 1, 0)

    score = np.sum(groups, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def keener_eigenvalues(reward_action: np.ndarray) -> np.array:
    """
    Name: Keener eigenvalues
    Desc: Action with the largest corresponding eigenvalue of the reward-action matrix is the winner.
    Source: http://www.9mail.de/m-schulze/votedesc.pdf
    """
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    eigenvalues, _ = np.linalg.eig(pairwise_matrix)
    winners = eigenvalues == np.max(eigenvalues)
    return winners

@voting
def prob_range_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Probabilistic range voting.
    Desc: For each action, reward values are summed.
          Normalized sums are returned as probabilities.
    Source: https://arxiv.org/abs/2006.06548, pg. 3
    """
    score = np.sum(reward_action, axis=1)
    score = normalize(score.reshape(-1, 1)).reshape(1, -1).squeeze()
    return score

@voting
def nash_lottery(reward_action: np.ndarray) -> np.array:
    """
    Name: Nash Lottery
    Desc: Gives probability range to maximize a log sum.
    Source: https://arxiv.org/abs/2006.06548
    """
    def log_sum(pX):
        return -np.sum(np.log((np.sum((normalize(reward_action.T) * pX).T, axis=1))))

    probb = Bounds(lb=np.zeros(reward_action.shape[0]), ub=np.ones(reward_action.shape[0]))
    probc = LinearConstraint(np.ones(reward_action.shape[0]), lb=0., ub=1.)
    ps = minimize(log_sum, np.ones(reward_action.shape[0]), bounds=probb, constraints=probc).x
    winners = ps
    return winners

@voting
def small_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Small method
    Desc: For each action, pairwise elections are held with all other actions.
          Each action's score is the total of its wins, plus half the total of its draws.
          Action with largest numer of points wins.
          In case of ties, drop all candidates below the max score. Repeat as necessary.
    Source: https://accuratedemocracy.com/download/software/rbvote/desc.html
    """
    red_reward_action = copy.deepcopy(reward_action)
    winners = np.ones(reward_action.shape[0])
    winners[0] = 1.0
    while np.count_nonzero(winners) > 1:
        score = pairwise_victory_score(red_reward_action, 1.0, 0.5)
        winners = score == np.max(score)
        red_reward_action[~winners] = -np.inf
    return winners

@voting
def first_preference_copeland(reward_action: np.ndarray) -> np.array:
    """
    Name: First preference Copeland
    Desc: Each action is assigned a penalty: the sum of first preferences of the candidates that beat the candidate pairwise.
          The action with the least penalty wins.
    Source: https://electowiki.org/wiki/First_preference_Copeland
    """
    half_action_shape = reward_action.shape[0] / 2.
    score = np.zeros(reward_action.shape[0])
    for action1 in range(reward_action.shape[0] - 1):
        for action2 in range(action1 + 1, reward_action.shape[0]):
            win_count = np.count_nonzero(np.greater(reward_action[action1], reward_action[action2]))
            if win_count < half_action_shape:
                score[action1] += np.count_nonzero(reward_action[action2] == np.max(reward_action, axis=0))
            elif win_count > half_action_shape:
                score[action2] += np.count_nonzero(reward_action[action1] == np.max(reward_action, axis=0))
    winners = score == np.min(score)
    return winners

@voting
def dodgson_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Dodgson method
    Desc: For each action, pairwise elections are held with all other actions.
          The candidate with the smallest sum of margins of defeat (largest sum of margins of victory) wins.
    Source: https://accuratedemocracy.com/download/software/rbvote/desc.html
    """
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    score = np.sum(pairwise_matrix - pairwise_matrix.T, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def simpson_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Simpson method
    Desc: For each action, pairwise elections are held with all other actions.
          The candidate with the smallest maximum margin of defeat (largest minimum margin of victory) wins.
    Source: https://accuratedemocracy.com/download/software/rbvote/desc.html
    """
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    score = np.min(pairwise_matrix - pairwise_matrix.T, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def reynaud_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Reynaud method
    Desc: For each action, pairwise elections are held with all other actions.
          The candidate with the largest single pairwise defeat (smallest single pairwise victory) is successively eliminated.
    Source: https://accuratedemocracy.com/download/software/rbvote/desc.html
    """
    red_reward_action = copy.deepcopy(reward_action)
    winners = np.ones(reward_action.shape[0], dtype='bool')
    score = np.array([0., 1.])
    while np.count_nonzero(winners) > 1:
        pairwise_matrix = pairwise_defeat_matrix(red_reward_action)
        score = np.min(pairwise_matrix - pairwise_matrix.T, axis=1)
        min_score = score == np.min(np.ma.MaskedArray(score, mask = ~winners))
        winners[min_score] = False
        red_reward_action[~winners] = -np.inf
    return winners

@voting
def minimax_condorcet_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Minimax Condorcet method
    Desc: For each action, pairwise elections are held with all other actions.
          The candidate with the largest number of votes in their worst matchup wins.
    Source: https://en.wikipedia.org/wiki/Minimax_Condorcet_method
    """
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    score = np.min(pairwise_matrix, axis=1)
    winners = score == np.max(score)
    return winners

@voting
def mmpo(reward_action: np.ndarray) -> np.array:
    """
    Name: MinMax (Pairwise Opposition), MMPO
    Desc: For each action, pairwise elections are held with all other actions.
          The candidate with the smallest number of votes for other candidates wins.
    Source: https://votingmethods.net/trunc/
    """
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    score = np.sum(pairwise_matrix, axis=0)
    winners = score == np.min(score)
    return winners

@voting
def brbo(reward_action: np.ndarray) -> np.array:
    """
    Name: Best response to best opposition, BRBO
    Desc: For each action, pairwise elections are held with all other actions.
          The candidate with the largest number of wins in a match against the action with the largest number of preferences against it wins.
    Source: https://votingmethods.net/trunc/
    """
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    score = np.argmax(pairwise_matrix, axis=0)
    max_score = -1.
    for i in range(reward_action.shape[0]):
        max_score = np.max([max_score, pairwise_matrix[i, score[i]]])
    winners = score == max_score
    return winners

@voting
def cross_max(reward_action: np.ndarray) -> np.array:
    """
    Name: Cross Max
    Desc: For each action, pairwise elections are held with all other actions.
          The candidate with the largest pairwise support wins.
    Source: https://votingmethods.net/trunc/
    """
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    winners = (pairwise_matrix == np.max(pairwise_matrix)).any(axis=1)
    return winners

@voting
def contingent_vote(reward_action: np.ndarray) -> np.array:
    """
    Name: Contingent vote, Top-Two Runoff, TTR
    Desc: Each reward gives an ordering of its preferences.
          If an action has a majority of first choices, it is chosen.
          If not, votes of the eliminated candidates are transferred to their next choices.
    Source: https://en.wikipedia.org/wiki/Contingent_vote
    """
    max_rank = reward_action.shape[0] - 1
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    score = np.sum(order == max_rank, axis=1)
    if (score > max_rank / 2.).any():
        winners = score == np.max(score)
    else:
        winners = (score == np.max(score)) | (score == np.sort(score)[-2])
        if np.count_nonzero(winners) > 2:
            return winners
        else:
            for ord in order.T:
                if ~np.isin(max_rank, ord[winners]):
                    score[ord==max_rank] += 1
        winners = score == np.max(score)
    return winners

@voting
def sri_lankan_contingent_vote(reward_action: np.ndarray) -> np.array:
    """
    Name: Sri Lankan contingent vote 
    Desc: Each reward gives an ordering of its first three preferences.
          If an action has a majority of first choices, it is chosen.
          If not, votes of the eliminated candidates are transferred to their next choices.
    Source: https://en.wikipedia.org/wiki/Contingent_vote#Sri_Lankan_contingent_vote
    """
    max_rank = reward_action.shape[0] - 1
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    score = np.sum(order == max_rank, axis=1)
    if (score > max_rank / 2.).any():
        winners = score == np.max(score)
    else:
        winners = (score == np.max(score)) | (score == np.sort(score)[-2])
        if np.count_nonzero(winners) > 2:
            return winners
        else:
            for ord in order.T:
                if ~np.isin(max_rank, ord[winners]):
                    score[(ord == max_rank) and (ord > max_rank - 3)] += 1
        winners = score == np.max(score)
    return winners

@voting
def supplementary_vote(reward_action: np.ndarray) -> np.array:
    """
    Name: Supplementary vote 
    Desc: Each reward gives an ordering of its first two preferences.
          If an action has a majority of first choices, it is chosen.
          If not, votes of the eliminated candidates are transferred to their next choices.
    Source: https://en.wikipedia.org/wiki/Contingent_vote#Supplementary_vote
    """
    max_rank = reward_action.shape[0] - 1
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    score = np.sum(order == max_rank, axis=1)
    if (score > max_rank / 2.).any():
        winners = score == np.max(score)
    else:
        winners = (score == np.max(score)) | (score == np.sort(score)[-2])
        if np.count_nonzero(winners) > 2:
            return winners
        else:
            for ord in order.T:
                if ~np.isin(max_rank, ord[winners]):
                    score[(ord == max_rank) and (ord > max_rank - 2)] += 1
        winners = score == np.max(score)
    return winners

@voting
def distributed_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Distributed voting
    Desc: Each ballot is normalized.
          Candidates with smallest sum of points received are successively eliminated.
    Source: https://electowiki.org/wiki/Distributed_Voting
    """
    init_reward_action = copy.deepcopy(reward_action)
    red_reward_action = copy.deepcopy(reward_action)
    score = np.array([0., 1.])
    while np.unique(score).shape[0] > 1:
        red_reward_action = red_reward_action * 100 / np.sum(red_reward_action, axis=0)
        score = np.sum(red_reward_action, axis=1)
        min_index = score == np.min(score)
        red_reward_action = np.delete(red_reward_action, min_index, axis=0)
        reward_action = np.delete(reward_action, min_index, axis=0)

    winners = np.zeros(init_reward_action.shape[0], dtype=bool)
    for i in range(reward_action.shape[0]):
        winners = winners | (init_reward_action == reward_action[i]).all(axis=1)
    return winners

@voting
def kemeny_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Kemeny-Young rule, VoteFair popularity ranking, Maximum likelihood method, Median relation.
    Desc: Calculates sum of pairwise preferences for each possible ranking of actions.
          Ranking with maximum score is picked.
    Source: https://en.wikipedia.org/wiki/Kemeny_method
    """
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    score = np.zeros(np.math.factorial(reward_action.shape[0]))
    rankings = np.array(list(multiset_permutations(np.array(range(reward_action.shape[0])))))
    for ranking in range(rankings.shape[0]):
        for r in range(rankings.shape[1] - 1):
            score[ranking] += pairwise_matrix[rankings[ranking, r], rankings[ranking, r+1]]
    max_rank = rankings[score == np.max(score)][0]
    winners = np.zeros(reward_action.shape[0], dtype='bool')
    winners[max_rank] = True
    return winners

@voting
def maxparc(reward_action: np.ndarray) -> np.array:
    """
    Name: Maximal Partial Consensus (MaxParC)
    Desc: Interprets score as willingness to compromise (probabilistic)
    Source: https://arxiv.org/abs/2006.06548
    """
    reward_action_norm = normalize(reward_action)
    approved = np.zeros_like(reward_action_norm)
    n_rewards = reward_action_norm.shape[1]
    for reward in range(n_rewards):
        order = np.argsort(np.argsort(reward_action_norm[:, reward])[::-1], axis=0)
        ordered_reward = np.sort(reward_action_norm[:, reward])[::-1]
        ordered_thresholds = np.array(range(reward_action_norm.shape[0]))[::-1] * 1. / (reward_action_norm.shape[0] - 1.)
        for ind in range(order.shape[0]):
            if ordered_reward[ind] < ordered_thresholds[ind]:
                break
            else:
                approved[order[ind], reward] += 1
    
    approval_scores = np.sum(approved, axis=1)
    chosen_ballot = np.random.randint(reward_action_norm.shape[1])
    possible_variants = approved[:, chosen_ballot] > 0.

    if possible_variants.all() == False:
        possible_variants[np.random.randint(reward_action_norm.shape[0])] = True
    winners = (approval_scores == np.max(approval_scores[possible_variants])) & possible_variants
    return winners

@voting
def maxparc_approval(reward_action: np.ndarray) -> np.array:
    """
    Name: Maximal Partial Consensus Approval
    Desc: Interprets score as willingness to compromise (deterministic)
    Source: https://arxiv.org/abs/2006.06548
    """
    reward_action_norm = normalize(reward_action)
    approved = np.zeros_like(reward_action_norm)
    n_rewards = reward_action_norm.shape[1]
    for reward in range(n_rewards):
        order = np.argsort(np.argsort(reward_action_norm[:, reward])[::-1], axis=0)
        ordered_reward = np.sort(reward_action_norm[:, reward])[::-1]
        ordered_thresholds = np.array(range(reward_action_norm.shape[0]))[::-1] * 1. / (reward_action_norm.shape[0] - 1.)
        for ind in range(order.shape[0]):
            if ordered_reward[ind] < ordered_thresholds[ind]:
                break
            else:
                approved[order[ind], reward] += 1
    
    approval_scores = np.sum(approved, axis=1)
    winners = approval_scores == np.max(approval_scores)
    return winners

@voting
def maximal_lottery(reward_action: np.ndarray) -> np.array:
    """
    Name: Maximal Lottery
    Desc: Returns probabilistic distribution over ballots to maximize satisfaction.
    Source: https://en.wikipedia.org/wiki/Maximal_lotteries
    """
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    def lottery_satisfaction(ps):
        return -np.sum((pairwise_matrix.T * ps).T)
    
    probc = LinearConstraint(np.ones(reward_action.shape[0]), lb=0., ub=1.)
    probb = Bounds(lb=np.zeros(reward_action.shape[0]), ub=np.ones(reward_action.shape[0]))
    ps = minimize(lottery_satisfaction, np.ones(reward_action.shape[0]), bounds=probb, constraints=probc).x
    winners = ps
    return winners

@voting
def maximal_lottery_lottery(reward_action: np.ndarray) -> np.array:
    """
    Name: Maximal Lottery-lottery
    Desc: Returns probabilistic distribution over maximal lotteries to maximize satisfaction.
    Source: https://www.lesswrong.com/s/gnAaZtdwjDBBRpDmw/p/vwrNprXfEzeQ2cy3d
    """
    # Get lotteries
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    def lottery_satisfaction(ps):
        return -np.sum((pairwise_matrix.T * ps).T)
    probc = LinearConstraint(np.ones(reward_action.shape[0]), lb=0., ub=1.)
    probb = Bounds(lb=np.zeros(reward_action.shape[0]), ub=np.ones(reward_action.shape[0]))

    lott1 = minimize(lottery_satisfaction, np.ones(reward_action.shape[0]), method='SLSQP', bounds=probb, constraints=probc).x
    lott2 = minimize(lottery_satisfaction, np.ones(reward_action.shape[0]), method='COBYLA', bounds=probb, constraints=probc).x
    lott3 = minimize(lottery_satisfaction, np.ones(reward_action.shape[0]), method='COBYQA', bounds=probb, constraints=probc).x
    lott4 = minimize(lottery_satisfaction, np.ones(reward_action.shape[0]), method='trust-constr', bounds=probb, constraints=probc).x

    # Get lottery-lottery
    lott_mat = np.stack((lott1, lott2, lott3, lott4))
    def lottery_lottery_satisfaction(ps):
        return -np.sum((lott_mat.T * ps).T)
    probc4 = LinearConstraint(np.ones(4), lb=0., ub=1.)
    probb4 = Bounds(lb=np.zeros(4), ub=np.ones(4))

    lott_lott = minimize(lottery_lottery_satisfaction, np.ones(4), bounds=probb4, constraints=probc4).x

    # Get lottery-lottery winner
    chosen_lott = np.random.choice([0, 1, 2, 3], p=lott_lott).item()
    winners = lott_mat[chosen_lott]
    return winners

@voting
def fplossa(reward_action: np.ndarray, threshold: str="mean") -> np.array:
    """
    Name: FPLossA
    Desc: Choose first preference action.
          If it has any pairwise losses, choose approval winner.
    Source: https://votingmethods.net/trunc/
    """
    max_actions = np.argmax(reward_action, axis=0)
    votes = np.bincount(max_actions, minlength=reward_action.shape[0])
    winners = votes == np.max(votes)

    defeats = pairwise_defeat_matrix(reward_action)
    if (defeats[:, winners] != 0.).any():
        if threshold == "mean":
            comps = np.mean(reward_action, axis=0)
        elif threshold == "median":
            comps = np.median(reward_action, axis=0)
        elif threshold == "t3":
            comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 3. * 2.
        elif threshold == "t1":
            comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 3.
        elif threshold == "q3":
            comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 4. * 3.
        elif threshold == "q1":
            comps = reward_action.min(axis=0) + (reward_action.max(axis=0) - reward_action.min(axis=0)) / 4.
        else:
            comps = np.mean(reward_action, axis=0)
        points = np.greater_equal(reward_action, comps)
        score = np.count_nonzero(points, axis=1)
        winners = score == np.max(score)
    return winners

@voting
def moral_parliament(reward_action: np.ndarray) -> np.array:
    """
    Name: Moral Parliament
    Desc: Gives probabilities corresponding to plurality votes.
    Source: https://amirrorclear.net/files/the-parliamentary-approach-to-moral-uncertainty.pdf
    """
    max_actions = np.argmax(reward_action, axis=0)
    votes = np.bincount(max_actions, minlength=reward_action.shape[0])
    winners = votes
    return winners

@voting
def proportional_borda(reward_action: np.ndarray) -> np.array:
    """
    Name: Proportional Borda
    Desc: Gives probabilities corresponding to Borda rankings.
    Source: original VoQL paper (https://pref-voting.readthedocs.io/en/stable/probabilistic_methods.html#proportional-borda)
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    score = np.sum(order, axis=1)
    winners = score
    return winners

@voting
def mdd_fpp(reward_action: np.ndarray) -> np.array:
    """
    Name: Majority Defeat Disqualification FPP, MDD FPP
    Desc: Actions are ranked.
          Action with the most first wins, as long as a majority is not against them.
    Source: https://votingmethods.net/trunc/
    """
    max_actions = np.argmax(reward_action, axis=0)
    min_actions = np.argmin(reward_action, axis=0)
    votes_for = np.bincount(max_actions, minlength=reward_action.shape[0])
    votes_against = np.bincount(min_actions, minlength=reward_action.shape[0])
    winners = (votes_for == np.max(votes_for)) & (votes_against < reward_action.shape[0] / 2.)
    return winners

@voting
def rcipe(reward_action: np.ndarray) -> np.array:
    """
    Name: Ranked Choice Including Pairwise Elimination, RCIPE
    Desc: IRV, but if any remaining candidate loses pairwise to all remaining candidates, it is eliminated instead of the lowest scorer.
    Source: https://votingmethods.net/trunc/
    """
    reward_action_inds = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    red_reward_action = copy.deepcopy(reward_action)

    grades = reward_action_inds.shape[0] - 1
    winners = np.ones(reward_action_inds.shape[0], dtype='bool')
    curr_votes = np.count_nonzero(reward_action_inds == grades, axis=1)
    while np.unique(curr_votes).shape[0] > 1:
        # Check for lowest
        lowest_action = np.argmin(np.ma.MaskedArray(curr_votes, mask = ~winners))
        pairwise_matrix = pairwise_defeat_matrix(red_reward_action)
        defeat_matrix = pairwise_matrix - pairwise_matrix.T
        complete_pairwise_loss = np.count_nonzero(defeat_matrix < 0., axis=1) == (reward_action_inds.shape[1] - 1.)
        if complete_pairwise_loss.any() == True:
            lowest_action = complete_pairwise_loss
        
        # Remove lowest
        winners[lowest_action] = False
        red_reward_action[~winners] = -np.inf
        red_reward_action[~winners] = -1.
        red_reward_action[winners] = np.argsort(np.argsort(red_reward_action[winners], axis=0), axis=0)
        grades -= 1
        curr_votes = np.count_nonzero(reward_action_inds == grades, axis=1)
    return winners

@voting
def koth(reward_action: np.ndarray) -> np.array:
    """
    Name: King of the Hill, KotH
    Desc: Choose pairwise victor between first preference winner (FPW) and candidate with most preferences with a full majority pairwise win over FPW.
    Source: https://votingmethods.net/trunc/
    """
    winners = np.zeros(reward_action.shape[0])
    max_actions = np.argmax(reward_action, axis=0)
    votes = np.bincount(max_actions, minlength=reward_action.shape[0])
    
    fpw_winner = np.argmax(votes)
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    if (pairwise_matrix[:, fpw_winner] > (reward_action.shape[0] / 2.0)).all() != False:
        pair_winner = np.max(votes[pairwise_matrix[:, fpw_winner] > (reward_action.shape[0] / 2.0)])
        pair_winner = (votes == pair_winner)[0]
        
        if np.count_nonzero(reward_action[fpw_winner] > reward_action[pair_winner]) > (reward_action.shape[1] / 2.0):
            winners[fpw_winner] = 1.0
        else:
            winners[pair_winner] = 1.0
    else:
        winners[fpw_winner] = 1.0
    return winners

@voting
def voice_of_reason(reward_action: np.ndarray) -> np.array:
    """
    Name: Voice of Reason
    Desc: Each reward receives one point for every pairwise preference of theirs which agrees with the overall preference among the rewards.
          Among the rewards with the highest score, elect the first preference winner.
    Source: https://votingmethods.net/trunc/
    """
    pairwise_matrix = pairwise_defeat_matrix(reward_action)
    preference_matrix = pairwise_matrix > pairwise_matrix.T
    rewards_score = np.zeros(reward_action.shape[1])
    for reward in range(reward_action.shape[1]):
        for action1 in range(reward_action.shape[0]):
            for action2 in range(reward_action.shape[0]):
                if action1 < action2:
                    if (preference_matrix[action1, action2] and reward_action[action1, reward] > reward_action[action2, reward])\
                    or (preference_matrix[action2, action1] and reward_action[action1, reward] < reward_action[action2, reward]):
                        rewards_score[reward] += 1
                else:
                    continue
    deciders = rewards_score == np.max(rewards_score)
    max_actions = np.argmax(reward_action[:, deciders], axis=0)
    votes = np.bincount(max_actions, minlength=reward_action.shape[0])
    winners = votes == np.max(votes)
    return winners

@voting
def tacc(reward_action: np.ndarray) -> np.array:
    """
    Name: Total Approval Chain Climbing
    Desc: Start with an empty set.
          Starting with the least approved action, evaluate whether the action pairwise beats all actions currently in the set. If so, add it to the set.
          Elect the last action who can be added.
    Source: https://votingmethods.net/trunc/
    """
    approval_set = np.zeros(reward_action.shape[0])
    max_actions = np.argmax(reward_action, axis=0)
    votes = np.bincount(max_actions, minlength=reward_action.shape[0])
    approval_set = votes == np.min(votes)
    last_added = np.argmin(votes)
    added = True
    while (np.count_nonzero(approval_set) < reward_action.shape[0]) and added:
        added = False
        for i in range(reward_action.shape[0]):
            if ~approval_set[i]:
                if (np.count_nonzero(reward_action[i] > reward_action[approval_set], axis=1) > (reward_action.shape[1] / 2.)).all():
                    approval_set[i] = 1.
                    last_added = i
                    added = True
    winners = np.zeros(reward_action.shape[0])
    winners[last_added] = 1.
    return winners

@voting
def dsc(reward_action: np.ndarray) -> np.array:
    """
    Name: Descending Solid Coalitions, DSC (Descending Acquiescing Coalitions, DAC for not strictly larger)
    Desc: Given n actions, there are 2^n-1 unique sets of actions.
          Each set receives a score, equal to the number of rewards who rank each action within that set strictly higher than every action outside that set.
          Assess each set starting with the highest score proceeding to the lowest.
          Every action not in the current set is disqualified from winning, unless this disqualifies all remaining actions, in which case the set has no effect.
          The last action remaining (i.e. never disqualified) is chosen.
    Source: https://votingmethods.net/trunc/
    """
    sets = np.zeros(int(np.power(2, reward_action.shape[0]) - 1.))
    rankings = np.argsort(np.argsort(reward_action, axis=0), axis=0)
    rev_ranks = np.array(range(reward_action.shape[0])[::-1])
    ranks = np.array(range(reward_action.shape[0]))
    for reward in range(reward_action.shape[1]):
        which_action = np.zeros(reward_action.shape[0], dtype='bool')
        for rank in rev_ranks:
            which_action = which_action | (rankings[:, reward] == rank)
            sets[int(np.sum(np.power(2.0, ranks[which_action])) - 1.)] += 1
    
    excluded_actions = np.zeros(reward_action.shape[0], dtype='bool')
    set_order = np.argsort(sets)[::-1]
    for i in set_order:
        bin_repr = np.binary_repr(i + 1, width = reward_action.shape[0])[::-1]
        temp_excluded_actions = copy.deepcopy(excluded_actions)
        for num in range(reward_action.shape[0]):
            if bin_repr[num] == '0':
                temp_excluded_actions[num] = True
        if np.count_nonzero(~temp_excluded_actions) == 0:
            continue
        else:
            excluded_actions = copy.deepcopy(temp_excluded_actions)
    winners = ~excluded_actions
    return winners


# https://pref-voting.readthedocs.io/en/stable/index.html
@voting
def schulze_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Schulze Method
    Desc: Popular graph-based method
    Source: https://pref-voting.readthedocs.io/en/stable/margin_based_methods.html#beat-path
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0).T
    order = order.tolist()
    prof = Profile(order)
    winners = margin_based_methods.beat_path(prof)
    winner_array = np.zeros(reward_action.shape[0])
    winner_array[winners] = 1.
    return winner_array

# https://pref-voting.readthedocs.io/en/stable/index.html
@voting
def tidemans_alternative(reward_action: np.ndarray) -> np.array:
    """
    Name: Tideman's Alternative method, Alternative-Smith voting
    Desc: Top-cycle (graph)-based method
    Source: https://pref-voting.readthedocs.io/en/stable/iterative_methods.html#tidemans-alternative
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0).T
    order = order.tolist()
    prof = Profile(order)
    winners = iterative_methods.tideman_alternative_smith(prof)
    winner_array = np.zeros(reward_action.shape[0])
    winner_array[winners] = 1.
    return winner_array

# https://pref-voting.readthedocs.io/en/stable/index.html
@voting
def woodall(reward_action: np.ndarray) -> np.array:
    """
    Name: Woodall's method
    Desc: Smith set-based method
    Source: https://pref-voting.readthedocs.io/en/stable/iterative_methods.html#woodall
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0).T
    order = order.tolist()
    prof = Profile(order)
    winners = iterative_methods.woodall(prof)
    winner_array = np.zeros(reward_action.shape[0])
    winner_array[winners] = 1.
    return winner_array

# https://pref-voting.readthedocs.io/en/stable/index.html
@voting
def plurality_veto(reward_action: np.ndarray) -> np.array:
    """
    Name: Plurality veto
    Desc: Iterative plurality-based method
    Source: https://pref-voting.readthedocs.io/en/stable/iterative_methods.html#plurality-veto
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0).T
    order = order.tolist()
    prof = Profile(order)
    winners = iterative_methods.plurality_veto(prof)
    winner_array = np.zeros(reward_action.shape[0])
    winner_array[winners] = 1.
    return winner_array

# https://pref-voting.readthedocs.io/en/stable/index.html
@voting
def consensus_builder(reward_action: np.ndarray) -> np.array:
    """
    Name: Consensus Builder
    Desc: Score-based method
    Source: https://pref-voting.readthedocs.io/en/stable/iterative_methods.html#consensus-builder
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0).T
    order = order.tolist()
    prof = Profile(order)
    winners = iterative_methods.consensus_builder(prof)
    winner_array = np.zeros(reward_action.shape[0])
    winner_array[winners] = 1.
    return winner_array

# https://pref-voting.readthedocs.io/en/stable/index.html
@voting
def benhams_method(reward_action: np.ndarray) -> np.array:
    """
    Name: Benham's method
    Desc: Iterative Condorcet method
    Source: https://pref-voting.readthedocs.io/en/stable/iterative_methods.html#benham
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0).T
    order = order.tolist()
    prof = Profile(order)
    winners = iterative_methods.benham(prof)
    winner_array = np.zeros(reward_action.shape[0])
    winner_array[winners] = 1.
    return winner_array

# https://pref-voting.readthedocs.io/en/stable/index.html
@voting
def ranked_pairs(reward_action: np.ndarray) -> np.array:
    """
    Name: Ranked Pairs, Tideman's rule
    Desc: Graph-based method
    Source: https://pref-voting.readthedocs.io/en/stable/margin_based_methods.html#ranked-pairs
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0).T
    order = order.tolist()
    prof = Profile(order)
    winners = margin_based_methods.ranked_pairs(prof)
    winner_array = np.zeros(reward_action.shape[0])
    winner_array[winners] = 1.
    return winner_array

# https://pref-voting.readthedocs.io/en/stable/index.html
@voting
def river_method(reward_action: np.ndarray) -> np.array:
    """
    Name: River method
    Desc: Graph-based method
    Source: https://pref-voting.readthedocs.io/en/stable/margin_based_methods.html#river
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0).T
    order = order.tolist()
    prof = Profile(order)
    winners = margin_based_methods.river(prof)
    winner_array = np.zeros(reward_action.shape[0])
    winner_array[winners] = 1.
    return winner_array

# https://pref-voting.readthedocs.io/en/stable/index.html
@voting
def stable_voting(reward_action: np.ndarray) -> np.array:
    """
    Name: Stable Voting
    Desc: Graph-based method
    Source: https://pref-voting.readthedocs.io/en/stable/margin_based_methods.html#stable-voting
    """
    order = np.argsort(np.argsort(reward_action, axis=0), axis=0).T
    order = order.tolist()
    prof = Profile(order)
    winners = margin_based_methods.stable_voting(prof)
    winner_array = np.zeros(reward_action.shape[0])
    winner_array[winners] = 1.
    return winner_array

"""
CONTROL METHODS
"""

@voting
def sortition(reward_action: np.ndarray) -> np.array:
    """
    Name: Sortition
    Desc: Choose winners at random.
    Source: Ancient Greece
    """
    winners = 0
    while np.sum(winners) == 0:
        winners = np.random.randint(2, size=reward_action.shape[0])
    return winners

@voting
def random_ballot(reward_action: np.ndarray) -> np.array:
    """
    Name: Random ballot, Gibbard Random Dictator
    Desc: Randomly pick a reward.
          Choose action which maximizes the chosen reward.
    Source: N/A
    """
    ballot = np.random.randint(reward_action.shape[1])
    max_action = np.argmax(reward_action[:, ballot], axis=0)
    votes = np.bincount([max_action], minlength=reward_action.shape[0])
    winners = votes == np.max(votes)
    return winners

@voting
def random_pair(reward_action: np.ndarray) -> np.array:
    """
    Name: Random pair, Gibbard Random Pair
    Desc: Choose two actions at random.
          Action with largest pairwise preferences wins.
    Source: http://www.9mail.de/m-schulze/votedesc.pdf
    """
    top_two = [0, 0]
    while top_two[0] == top_two[1]:
        top_two = np.random.randint(reward_action.shape[0], size=2)
    first_winner = np.count_nonzero(reward_action[top_two[0]] > reward_action[top_two[1]]) > (reward_action.shape[0] / 2.)
    winners = np.zeros(reward_action.shape[0])
    if first_winner:
        winners[top_two[0]] = 1
    else:
        winners[top_two[1]] = 1
    return winners

@voting
def worst_winner(reward_action: np.ndarray, normalized: bool=False) -> np.array:
    """
    Name: Worst winner
    Desc: For each action, reward values are summed.
          Action with the lowest sum is picked.
    Source: http://www.9mail.de/m-schulze/votedesc.pdf
    """
    if normalized:
        reward_action_norm = normalize(reward_action)
    else:
        reward_action_norm = reward_action
    score = np.sum(reward_action_norm, axis=1)
    winners = score == np.min(score)
    return winners
