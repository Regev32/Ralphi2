# train_clique_on_file.py

import ast                                   # ← CHANGED: for parsing Python literals from lines.txt
import torch
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

from clique_env import CliqueEnv
from actor_critic_net import ActorCriticNet

device = torch.device("cpu")


def load_all_graphs_from_file(txt_path: str) -> list:
    """
    Read each non‐empty line of `txt_path`, parse it with ast.literal_eval,
    and return a list of 5‐partite graphs.
    """
    graphs = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = ast.literal_eval(line)   # ← CHANGED
            except Exception as e:
                raise RuntimeError(f"Failed to parse line as Python literal:\n{line}\nError: {e}")
            # Basic sanity check:
            if not (isinstance(g, list) and len(g) == 5 and all(isinstance(p, list) and len(p) == 2 for p in g)):
                raise ValueError(f"Line did not evaluate to a 5‐partition graph: {line}")
            graphs.append(g)
    if not graphs:
        raise RuntimeError(f"No valid graphs found in {txt_path}")
    return graphs


def train_actor_critic_on_file(
    graph_list: list,
    num_epochs: int = 3,              # ← CHANGED: how many passes over the file
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
    # (A) Build a dummy env from the first graph to get
    #     total_actions and observation_space
    dummy_env = CliqueEnv(graph_list[0], max_steps=max_steps_per_env)
    total_actions = dummy_env.total_actions
    obs_spaces = dummy_env.observation_space
    dummy_env.close()

    # (B) Instantiate a shared ActorCriticNet + optimizer
    net = ActorCriticNet(obs_spaces, total_actions, hidden_size=128).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    episode_counter = 0

    for epoch in range(1, num_epochs + 1):
        # Shuffle graphs each epoch if you want
        np.random.shuffle(graph_list)

        for graph_idx, graph in enumerate(graph_list, start=1):
            episode_counter += 1

            # (1) New environment for THIS graph
            env = CliqueEnv(graph, max_steps=max_steps_per_env)

            # (2) Single‐episode rollout
            obs_dict, _ = env.reset()
            done = False
            truncated = False

            log_probs = []
            values = []
            rewards = []

            while not (done or truncated):
                ws_t = torch.tensor(obs_dict["which_side"], dtype=torch.float32).unsqueeze(0).to(device)
                c1_t = torch.tensor(obs_dict["choice1"], dtype=torch.float32).unsqueeze(0).to(device)
                c2_t = torch.tensor(obs_dict["choice2"], dtype=torch.float32).unsqueeze(0).to(device)

                logits, value = net({"which_side": ws_t, "choice1": c1_t, "choice2": c2_t})
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

                next_obs, reward, done, truncated, _ = env.step(int(action.item()))

                log_probs.append(logprob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))

                obs_dict = next_obs

            # (3) Compute returns & advantages
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.cat(returns).detach()       # shape (T,)
            values_t = torch.cat(values).squeeze(-1)    # shape (T,)
            log_probs_t = torch.cat(log_probs)          # shape (T,)

            advantages = returns - values_t
            actor_loss  = -(log_probs_t * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss

            # (4) Single gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            env.close()

            # (5) Diagnostics
            if episode_counter % print_every == 0:
                avg_return = returns.mean().item()
                print(
                    f"Epoch {epoch:2d} | Graph {graph_idx:3d}/{len(graph_list):3d} "
                    f" (episode #{episode_counter:4d}) | "
                    f"avg_return ≃ {avg_return:.4f} | loss={loss.item():.4f}"
                )

    # (C) Save final weights
    torch.save(net.state_dict(), "clique_ac_model_allgraphs.pth")
    print(f"\n>>> Completed {episode_counter} episodes over {num_epochs} epochs. "
          f"Weights saved as clique_ac_model_allgraphs.pth <<<")

    return net


if __name__ == "__main__":
    txt_path = "../../data/typing/length10.txt"
    graphs = load_all_graphs_from_file(txt_path)
    print(f"Loaded {len(graphs)} graphs from {txt_path}")

    trained_net = train_actor_critic_on_file(
        graph_list=graphs,
        num_epochs=3,
        gamma=0.99,
        lr=1e-4,
        print_every=20,
        max_steps_per_env=30,
    )

