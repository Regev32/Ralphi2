# train_clique_on_file.py

import ast
import torch
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import logging
import json

from clique_AC.clique_env import CliqueEnv
from clique_AC.actor_critic_net import ActorCriticNet

device = torch.device("cpu")

# --- Configure logging ---
LOG_FILE_NAME = 'training_log.txt'

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler(LOG_FILE_NAME, encoding='utf-8')
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

if logger.hasHandlers():
    logger.handlers.clear()

# Add the file handler to the logger
logger.addHandler(file_handler)


def load_all_graphs_from_file(txt_path: str) -> list:
    """
    Read each non‐empty line of `txt_path`, parse it with ast.literal_eval,
    and return a list of 5‐partite graphs.
    """
    graphs = []
    # Removed initial print/log for "Loading graphs from..."
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = ast.literal_eval(line)
            except Exception as e:
                raise RuntimeError(f"Failed to parse line as Python literal:\n{line}\nError: {e}")
            if not (isinstance(g, list) and len(g) == 5 and all(isinstance(p, list) and len(p) == 2 for p in g)):
                raise ValueError(f"Line did not evaluate to a 5‐partition graph: {line}")
            graphs.append(g)
    if not graphs:
        raise RuntimeError(f"No valid graphs found in {txt_path}")
    return graphs


def train_actor_critic_on_file(
    graph_list: list,
    num_epochs: int = 3,
    gamma: float = 0.99,
    lr: float = 1e-4,
    print_every: int = 20,
    max_steps_per_env: int = 30,
):
    """
    Train over each graph in graph_list, for num_epochs passes.
    For each graph:
      1) Create a fresh CliqueEnv(graph, max_steps_per_env)
      2) Run exactly one episode (collect transitions)
      3) Compute losses & do one gradient update on the shared net.
    """
    dummy_env = CliqueEnv(graph_list[0], max_steps=max_steps_per_env)
    total_actions = dummy_env.total_actions
    obs_spaces = dummy_env.observation_space
    dummy_env.close()

    net = ActorCriticNet(obs_spaces, total_actions, hidden_size=128).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    episode_counter = 0

    for epoch in range(1, num_epochs + 1):
        np.random.shuffle(graph_list)

        for graph_idx, graph in enumerate(graph_list, start=1):
            episode_counter += 1

            env = CliqueEnv(graph, max_steps=max_steps_per_env)

            obs_dict, _ = env.reset()
            done = False
            truncated = False

            log_probs = []
            values = []
            rewards = []
            tokenizer = None
            with open("../graphs/indexes.json", "r") as f:
                tokenizer = json.load(f)
            while not (done or truncated):
                ws_t = torch.tensor(obs_dict["which_side"], dtype=torch.float32).unsqueeze(0).to(device)
                idxs = [tokenizer[a] for a in obs_dict["choice1"]]
                c1_t = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(device)
                idxs = [tokenizer[a] for a in obs_dict["choice2"]]
                c2_t = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(device)

                logits, value = net({"which_side": ws_t, "choice1": c1_t, "choice2": c2_t})
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

                next_obs, reward, done, truncated, _ = env.step(int(action.item()))

                log_probs.append(logprob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))

                obs_dict = next_obs

            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.cat(returns).detach()
            values_t = torch.cat(values).squeeze(-1)
            log_probs_t = torch.cat(log_probs)

            advantages = returns - values_t
            actor_loss  = -(log_probs_t * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            env.close()

            if episode_counter % print_every == 0:
                avg_return = returns.mean().item()
                # Use logger.info instead of print for the desired lines
                logger.info(f"Epoch {epoch:2d} | Graph {graph_idx:3d}/{len(graph_list):3d} (episode #{episode_counter:4d}) | avg_return ≃ {avg_return:.4f} | loss={loss.item():.4f}")
                print(f"Epoch {epoch:2d} | Graph {graph_idx:3d}/{len(graph_list):3d} (episode #{episode_counter:4d}) | avg_return ≃ {avg_return:.4f} | loss={loss.item():.4f}")
    # (C) Save final weights
    torch.save(net.state_dict(), "clique_ac_model_allgraphs.pth")
    # Removed final log for "Completed X episodes..."
    return net