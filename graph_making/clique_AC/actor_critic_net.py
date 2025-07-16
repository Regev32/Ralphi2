import json
import os

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
        # obs_spaces has length‐5 vectors for which_side, choice1, choice2
        self.seq_len = 5
        self.embed_dim = 20
        with open("../graphs/indexes.json", "r") as f:
            self.tokenizer = json.load(f)

        # assume tokenizer has been built elsewhere, here we pass in its vocab size
        vocab_size = len(self.tokenizer)
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)

        # now the flattened input size is:
        #  - which_side: 5 floats
        #  - choice1 embeddings: 5 × embed_dim
        #  - choice2 embeddings: 5 × embed_dim
        obs_dim = self.seq_len + 2 * self.seq_len * self.embed_dim

        self.shared_fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, total_actions)
        self.value_head  = nn.Linear(hidden_size, 1)

    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # ws: (B,5) floats
        ws = obs_dict["which_side"].float()

        # c1, c2: (B,5) long indices into the embedding
        c1_idx = obs_dict["choice1"].long()
        c2_idx = obs_dict["choice2"].long()

        # embed -> (B,5,embed_dim)
        c1_emb = self.embedding(c1_idx)
        c2_emb = self.embedding(c2_idx)

        # flatten sequence dims -> (B, 5*embed_dim)
        B = ws.size(0)
        c1_flat = c1_emb.view(B, -1)
        c2_flat = c2_emb.view(B, -1)

        # concat everything -> (B, obs_dim)
        x = torch.cat([ws, c1_flat, c2_flat], dim=1)

        h = self.shared_fc(x)               # (B, hidden_size)
        logits = self.policy_head(h)        # (B, total_actions)
        value  = self.value_head(h).squeeze(-1)  # (B,)
        return logits, value
