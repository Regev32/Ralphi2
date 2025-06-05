import torch
import torch.nn as nn
from typing import Tuple

class ActorCriticNet(nn.Module):
    """
    Takes in the observation‐dict from CliqueEnv and outputs:
      - policy_logits (size = total_actions)
      - state_value (scalar)
    """
    def __init__(self, obs_spaces, total_actions: int, hidden_size: int = 128):
        super().__init__()
        # We know obs_spaces is a Dict with 3 arrays of length=5 each
        # So the flattened input dim is 15
        obs_dim = 5 + 5 + 5

        self.shared_fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, total_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs_dict["which_side"], obs_dict["choice1"], obs_dict["choice2"] are Tensors of shape (B,5)
        ws = obs_dict["which_side"].float()
        c1 = obs_dict["choice1"].float()
        c2 = obs_dict["choice2"].float()
        x = torch.cat([ws, c1, c2], dim=1)   # shape (B, 15)
        h = self.shared_fc(x)               # shape (B, hidden_size)
        logits = self.policy_head(h)        # shape (B, total_actions)
        value = self.value_head(h).squeeze(-1)  # shape (B,)
        return logits, value
