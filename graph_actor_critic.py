# graph_actor_critic.py
# Custom Gym Environment + A2C Training over All Patients from CSV

import os
import json
import pickle
import networkx as nx
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import logging
import random
import itertools           # for combinations
import matplotlib.pyplot as plt

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Always resolve paths relative to this file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def parse_groups(hla_string):
    """
    Split on '^' then '+'; keep '/' inside each chunk.
    Returns a list of group-strings.
    """
    groups = []
    for locus_part in hla_string.split('^'):
        for chunk in locus_part.split('+'):
            g = chunk.strip()
            if g:
                groups.append(g)
    return groups


def build_subgraph_for_hla(groups, G_full):
    """
    Given group-strings (with slashes) and the full graph,
    build an unweighted subgraph whose nodes are exactly those groups,
    inheriting and summing any edges present in G_full.
    """
    var2grp = {v: grp for grp in groups for v in grp.split('/')}
    sub = nx.Graph()
    sub.add_nodes_from(groups)
    for grp in groups:
        for var in grp.split('/'):
            if var not in G_full:
                continue
            for nbr, data in G_full[var].items():
                if nbr in var2grp:
                    tgt = var2grp[nbr]
                    if tgt == grp:
                        continue
                    w = data.get('weight', 1.0)
                    if sub.has_edge(grp, tgt):
                        sub[grp][tgt]['weight'] += w
                    else:
                        sub.add_edge(grp, tgt, weight=w)
    return sub


def brute_force_best(subG):
    """
    Brute-force the maximum intra-clique weight on subG.
    """
    nodes = list(subG.nodes())
    best  = -np.inf
    for assign in itertools.product([1, -1], repeat=len(nodes)):
        w = 0.0
        for sign in (1, -1):
            clique = [nodes[i] for i, a in enumerate(assign) if a == sign]
            for u, v in itertools.combinations(clique, 2):
                if subG.has_edge(u, v):
                    w += subG[u][v]['weight']
        best = max(best, w)
    return best


class GraphPartitionEnv(gym.Env):
    """
    Gym environment for splitting allele-nodes into two cliques.
    - The *world* is a subgraph of fixed groups.
    - Rewards always come from the full graph but are normalized.
    """
    metadata = {"render.modes": []}
    _cached_full_graph = None

    def __init__(self, hla_string, cfg):
        super().__init__()

        # Load or reuse the full graph
        if GraphPartitionEnv._cached_full_graph is None:
            pkl = os.path.join(BASE_DIR, cfg['graph_path'],
                               cfg['donors_file'] + '.pkl')
            logger.info(f"Loading full HLA graph from {pkl}")
            GraphPartitionEnv._cached_full_graph = pickle.load(open(pkl, 'rb'))
        self.G_full = GraphPartitionEnv._cached_full_graph

        # Parse groups and build subgraph
        parsed     = parse_groups(hla_string)
        self.G_sub = build_subgraph_for_hla(parsed, self.G_full)

        # Fixed world of groups
        self.groups   = parsed
        self.num_loci = len(self.groups)

        # Spaces
        self.action_space = gym.spaces.Discrete(self.num_loci * 4)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=max(1.0, float(cfg.get('alpha', 1.0))),
            shape=(4 * self.num_loci,),
            dtype=np.float32
        )

        # Hyperparams
        self.max_steps = cfg.get('max_steps', 2000)
        self.alpha     = cfg.get('alpha', 1.0)

        self.reset()

    def reset(self):
        # Random assignment and allele picks
        self.assign = np.random.choice([1, -1], size=(self.num_loci, 2))
        self.alleles = np.empty((self.num_loci, 2), dtype=object)
        for i, grp in enumerate(self.groups):
            opts = grp.split('/')
            self.alleles[i, 0] = random.choice(opts)
            self.alleles[i, 1] = random.choice(opts)

        self.step_count  = 0
        self.prev_weight = self._calc_weight()
        # compute normalization constant
        self.max_possible = brute_force_best(self.G_sub) or 1.0
        return self._obs()

    def _obs(self):
        assign_flat = self.assign.flatten().astype(np.float32)
        allele_idx  = []
        for i, grp in enumerate(self.groups):
            opts = grp.split('/')
            allele_idx.append(opts.index(self.alleles[i, 0]))
            allele_idx.append(opts.index(self.alleles[i, 1]))
        return np.concatenate([assign_flat, np.array(allele_idx, dtype=np.float32)])

    def _calc_weight(self):
        weight = 0.0
        for sign in (1, -1):
            clique = []
            for i in range(self.num_loci):
                for j in (0, 1):
                    if self.assign[i, j] == sign:
                        clique.append(self.alleles[i, j])
            for u, v in itertools.combinations(clique, 2):
                if self.G_full.has_edge(u, v):
                    weight += self.G_full[u][v]['weight']
        return weight

    def step(self, action):
        locus = action // 4
        op    = action % 4

        if op == 0:
            self.assign[locus] *= -1

        opts = self.groups[locus].split('/')
        if op in (1, 3):
            self.alleles[locus, 0] = random.choice(opts)
        if op in (2, 3):
            self.alleles[locus, 1] = random.choice(opts)

        current_weight = self._calc_weight()
        raw_delta      = current_weight - self.prev_weight
        # normalize reward
        reward         = self.alpha * (raw_delta / self.max_possible)
        self.prev_weight = current_weight

        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        return self._obs(), reward, done, {}


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        b = self.shared(x)
        return self.actor(b), self.critic(b).squeeze(-1)


def train(cfg):
    donors_csv = os.path.join(BASE_DIR,
                              cfg['donors_folder'],
                              cfg['donors_file'] + '.csv')
    skip_header = cfg.get('first_row_headers', False)
    hla_list = []
    with open(donors_csv) as f:
        if skip_header: next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 1 and parts[1].strip():
                hla_list.append(parts[1].strip())

    logger.info(f"Loaded {len(hla_list)} HLA entries from {donors_csv}")

    episodes = cfg.get('episodes', 500)
    lr       = cfg.get('lr', 3e-4)
    gamma    = cfg.get('gamma', 0.99)
    save_path= cfg.get('save_model', 'models/actor_critic.pth')
    logdir   = cfg.get('logdir', 'runs')

    # build a fixed example HLA for consistent dims
    loci_json = os.path.join(BASE_DIR,
                             cfg['graph_path'],
                             cfg['donors_file'] + '_loci.json')
    with open(loci_json) as lf:
        alleles_by_locus = json.load(lf)
    example = []
    for locus in cfg['allowed_loci']:
        vals = alleles_by_locus[locus]
        a1   = vals[0]
        a2   = vals[1] if len(vals) > 1 else vals[0]
        example.append(f"{a1}+{a2}")
    example_hla = "^".join(example)

    dummy      = GraphPartitionEnv(example_hla, cfg)
    state_dim  = dummy.observation_space.shape[0]
    action_dim = dummy.action_space.n
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer    = SummaryWriter(logdir) if TB_AVAILABLE else None
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rewards_history = []
    for ep in range(1, episodes + 1):
        hla_string = random.choice(hla_list)
        env        = GraphPartitionEnv(hla_string, cfg)
        state      = env.reset()

        log_probs, values, rewards = [], [], []
        done = False
        while not done:
            st    = torch.from_numpy(state).float().to(device)
            probs, value = model(st)
            dist   = Categorical(probs)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            state = next_state

        returns, R = [], 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns    = torch.tensor(returns).to(device)
        values     = torch.stack(values)
        log_probs  = torch.stack(log_probs)
        advantages = returns - values

        actor_loss  = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss        = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        rewards_history.append(total_reward)

        if ep % 10 == 0:
            last10      = rewards_history[-10:]
            max_last10  = np.max(last10)
            logger.info(f"Episode {ep}/{episodes} | Max(last 10): {max_last10:.2f}")

    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    plt.figure()
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Curve')
    plt.tight_layout()
    out_plot = os.path.join(os.path.dirname(save_path), 'training_rewards.png')
    plt.savefig(out_plot)
    logger.info(f"Training curve saved to {out_plot}")
    if writer:
        writer.close()


if __name__ == "__main__":
    cfg = json.load(open(os.path.join(BASE_DIR, 'config.json')))
    train(cfg)
