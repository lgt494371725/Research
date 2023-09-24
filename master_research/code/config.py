"""
heuristic: TSP,MST,agg_h, None
start: give the pos responding to n_agent
"""
optional_params = {"f_weight": [1],
                   "IW": [False],
                   "WR": [False],
                   "n_agent": [4],
                   "heuristic": ["agg_h"],
                   "hash_structure": [True],
                   "pre-pruning": [True]
                   }
# map = np.array([[0, 0, 0, 0, 0],
#                 [1, 1, 0, 1, 1],
#                 [0, 0, 0, 0, 0],
#                 [0, 1, 1, 1, 0],
#                 [0, 1, 0, 0, 0]])
# start = [(0, 0), (0, 4)]
