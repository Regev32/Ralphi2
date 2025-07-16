import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Any, Tuple
import json
import pickle as pkl
import os

def calculate_clique_weight( alleles = None, conf_path = "../config.json"):
    with open(conf_path) as json_file:
        CONF = json.load(json_file)
    graph_path = os.path.join("..", CONF['graph_path'], CONF['donors_file'] + '.pkl')
    with open(graph_path, 'rb') as f:
        G = pkl.load(f)
    return sum(
        G[u][v]['weight']
        for i, u in enumerate(alleles)
        for j, v in enumerate(alleles)
        if i < j and G.has_edge(u, v)
    )


class CliqueEnv(gym.Env):
    """
    Custom Gymnasium environment for two-clique selection over a 5-partite graph.
    Reward at each step is normalized by the total weight of all nodes in the subgraph.

    Each partition i has exactly two lists: graph_partitions[i][0] and graph_partitions[i][1].
    which_side[i] ∈ {0,1} indicates which list goes to clique 1 (the other list goes to clique 2).
    choice1[i], choice2[i] select a node-index within each assigned list.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        graph_partitions: List[List[List[Any]]],
        max_steps: int = 50,
    ):
        super().__init__()
        assert len(graph_partitions) == 5, "Need exactly 5 partitions (length=5)."

        self.graph_partitions = graph_partitions
        self.num_partitions = 5
        self.max_steps = max_steps

        # Internal state: which list from each partition goes to clique1, and which node-indices are chosen
        self.which_side = np.zeros(self.num_partitions, dtype=np.int64)
        self.choice1    = [None] * self.num_partitions
        self.choice2    = [None] * self.num_partitions
        self._step_count = 0
        self._prev_sum   = 0.0
        self._total_graph_weight = 1.0  # will be overwritten in reset()

        # Precompute maximum number of “atomic actions” in each partition:
        #   = 1 (flip) + len(list0) + len(list1)
        self.max_actions_per_partition = []
        for i in range(self.num_partitions):
            n0 = len(self.graph_partitions[i][0])
            n1 = len(self.graph_partitions[i][1])
            self.max_actions_per_partition.append(1 + n0 + n1)

        # Build a single Discrete action space by concatenating each partition’s block
        self.base_offsets = np.cumsum([0] + self.max_actions_per_partition[:-1], dtype=np.int64)
        self.total_actions = int(self.base_offsets[-1] + self.max_actions_per_partition[-1])
        self.action_space = spaces.Discrete(self.total_actions)

        # Observation: a Dict of three length-5 arrays
        obs_spaces = {
            "which_side": spaces.MultiBinary(self.num_partitions),
            "choice1":    spaces.MultiDiscrete(
                              [ len(self.graph_partitions[i][0]) + len(self.graph_partitions[i][1])
                                for i in range(self.num_partitions) ]
                          ),
            "choice2":    spaces.MultiDiscrete(
                              [ len(self.graph_partitions[i][0]) + len(self.graph_partitions[i][1])
                                for i in range(self.num_partitions) ]
                          ),
        }
        self.observation_space = spaces.Dict(obs_spaces)
        with open("../graphs/indexes.json", "r") as f:
            self.tokenizer = json.load(f)

    def _get_obs(self) -> dict:
        return {
            "which_side": self.which_side.copy(),
            "choice1":    self.choice1.copy(),
            "choice2":    self.choice2.copy(),
        }

    def _build_cliques(self) -> Tuple[List[Any], List[Any]]:
        """
        From (which_side, choice1, choice2), build the two lists of 5 nodes each.
        """
        clique1 = []
        clique2 = []
        tokenizer = None

        for i in range(self.num_partitions):
            side = int(self.which_side[i])      # 0 or 1
            list_c1 = self.graph_partitions[i][side]
            list_c2 = self.graph_partitions[i][1 - side]

            idx1 = int(self.tokenizer[self.choice1[i]])
            idx2 = int(self.tokenizer[self.choice2[i]])

            # Clamp to valid indices
            idx1 = np.clip(idx1, 0, len(list_c1) - 1)
            idx2 = np.clip(idx2, 0, len(list_c2) - 1)

            clique1.append(list_c1[idx1])
            clique2.append(list_c2[idx2])

        return clique1, clique2

    def _compute_sum(self) -> float:
        """
        Compute (weight of clique1) + (weight of clique2).
        """
        c1, c2 = self._build_cliques()
        return float(calculate_clique_weight(c1) + calculate_clique_weight(c2))

    def _compute_total_graph_weight(self) -> float:
        """
        Collect ALL nodes from every list in every partition and call calculate_clique_weight on them.
        That becomes the normalizing constant.
        """
        all_nodes = []
        for i in range(self.num_partitions):
            # Extend by both lists in partition i
            all_nodes.extend(self.graph_partitions[i][0])
            all_nodes.extend(self.graph_partitions[i][1])
        return float(calculate_clique_weight(all_nodes))

    def reset(
            self,
            *,
            seed: int = None,
            options: dict = None
    ) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        # 1) Randomly assign which_side[i] ∈ {0,1}
        self.which_side = self.np_random.integers(
            0, 2, size=(self.num_partitions,), dtype=np.int64
        )

        # 2) For each partition i, pick a random allele for clique1 & clique2
        for i in range(self.num_partitions):
            side = int(self.which_side[i])
            list_c1 = self.graph_partitions[i][side]
            list_c2 = self.graph_partitions[i][1 - side]

            # draw random indices into each list
            idx1 = int(self.np_random.integers(0, len(list_c1)))
            idx2 = int(self.np_random.integers(0, len(list_c2)))

            # store the actual allele strings
            self.choice1[i] = list_c1[idx1]
            self.choice2[i] = list_c2[idx2]

        self._step_count = 0

        # 3) Compute both: previous sum and total subgraph sum for normalization
        self._prev_sum = self._compute_sum()
        self._total_graph_weight = self._compute_total_graph_weight()

        obs = self._get_obs()
        return obs, {}

    def step(
            self,
            action: int
    ) -> Tuple[dict, float, bool, bool, dict]:
        self._step_count += 1

        # (A) Decode which partition i this action belongs to
        partition_i = None
        local_id = None
        for i in range(self.num_partitions):
            b = int(self.base_offsets[i])
            m = int(self.max_actions_per_partition[i])
            if b <= action < b + m:
                partition_i = i
                local_id = action - b
                break
        if partition_i is None:
            raise ValueError(f"Action {action} out of range [0..{self.total_actions - 1}]")
        i = partition_i

        # (B) Grab current lists
        side = int(self.which_side[i])
        list_c1 = self.graph_partitions[i][side]
        list_c2 = self.graph_partitions[i][1 - side]

        # (C) Decode local_id
        if local_id == 0:
            # FLIP which_side[i]
            self.which_side[i] = 1 - side
            new_side = int(self.which_side[i])
            new_list_c1 = self.graph_partitions[i][new_side]
            new_list_c2 = self.graph_partitions[i][1 - new_side]

            # If the old allele is no longer in the new list, re‑sample
            if self.choice1[i] not in new_list_c1:
                idx = int(self.np_random.integers(0, len(new_list_c1)))
                self.choice1[i] = new_list_c1[idx]
            if self.choice2[i] not in new_list_c2:
                idx = int(self.np_random.integers(0, len(new_list_c2)))
                self.choice2[i] = new_list_c2[idx]

        else:
            # A non‑flip action picks from the *current* lists
            # local_id in [1 .. len(list_c1)] → set choice1
            if 1 <= local_id <= len(list_c1):
                idx = local_id - 1
                self.choice1[i] = list_c1[idx]
            # local_id in [1+len(list_c1) .. 1+len(list_c1)+len(list_c2)-1] → set choice2
            elif 1 + len(list_c1) <= local_id < 1 + len(list_c1) + len(list_c2):
                idx = local_id - (1 + len(list_c1))
                self.choice2[i] = list_c2[idx]
            # else: ignore

        # (D) Compute reward
        new_sum = self._compute_sum()
        delta = new_sum - self._prev_sum
        reward = delta / max(self._total_graph_weight, 1e-12)
        self._prev_sum = new_sum

        # (E) Termination
        terminated = False
        truncated = (self._step_count >= self.max_steps)

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, {}

    def render(self, mode="human"):
        c1, c2 = self._build_cliques()
        total = self._compute_sum()
        print(f"Step {self._step_count}:")
        print("  Clique 1:", c1)
        print("  Clique 2:", c2)
        print("  Total sum:", total)

    def close(self):
        pass
