# train_clique.py

import torch
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

from clique_env import CliqueEnv
from actor_critic_net import ActorCriticNet

device = torch.device("cpu")

def train_actor_critic(
    env: CliqueEnv,
    net: ActorCriticNet,
    num_episodes: int = 500,
    gamma: float = 0.99,
    lr: float = 1e-4,
    print_every: int = 50,
):
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for ep in range(1, num_episodes + 1):
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

        # Compute returns & advantages
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

        if ep % print_every == 0:
            avg_return = returns.mean().item()
            print(f"Episode {ep:4d} | avg_return ≃ {avg_return:.2f} | loss={loss.item():.4f}")

if __name__ == "__main__":
    # This used to train on a single hard‐coded graph:
    graph_partitions = [
        [['a','b'], ['c','d','e']],
        [['f'], ['i','j','k']],
        [['l','m'], ['n']],
        [['o','p','q'], ['r']],
        [['s','t','u'], ['v']],
    ]

    env = CliqueEnv(graph_partitions, max_steps=30)
    total_actions = env.total_actions
    net = ActorCriticNet(env.observation_space, total_actions, hidden_size=128).to(device)

    train_actor_critic(env, net, num_episodes=500, gamma=0.99, lr=1e-4, print_every=50)
    torch.save(net.state_dict(), "clique_ac_model.pth")
