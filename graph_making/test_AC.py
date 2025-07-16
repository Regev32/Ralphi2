#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import ast
import random
from itertools import groupby, product

import numpy as np
import torch
from tqdm import tqdm

from clique_AC.clique_env import CliqueEnv
from clique_AC.actor_critic_net import ActorCriticNet

def load_hla_lines(path: str):
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # literal_eval will parse "[[['A*02:01'], ...]]" into Python lists
            nested = ast.literal_eval(line)
            results.append(nested)
    return results

def create_hla(conf_path):
    with open(conf_path, 'r') as f:
        cfg = json.load(f)
    loci_path = os.path.join('..', cfg['donors_folder'], f"length10{cfg['donors_file']}_TEST.txt")
    hla_lines = load_hla_lines(loci_path)
    return hla_lines

# ---- Weight Computations ----
def calculate_clique_weight(G, alleles):
    return sum(
        G[u][v]['weight']
        for i, u in enumerate(alleles)
        for j, v in enumerate(alleles)
        if i < j and G.has_edge(u, v)
    )

def generate_hla_combinations(G, parsed_hla):
    per_locus = []
    for locus in parsed_hla:
        G0, G1 = locus
        pairs = [(a, b) for a in G0 for b in G1] + [(b, a) for a in G0 for b in G1]
        per_locus.append(pairs)

    max_w = 0.0
    for combo in product(*per_locus):
        h1 = [p[0] for p in combo]
        h2 = [p[1] for p in combo]
        w = calculate_clique_weight(G, h1) + calculate_clique_weight(G, h2)
        if w > max_w:
            max_w = w
    return max_w

# ---- Imputation Methods ----

def impute_hla(parsed_hla):
    env = CliqueEnv(parsed_hla, max_steps=MAX_STEPS)
    net = ActorCriticNet(env.observation_space, env.total_actions, hidden_size=128).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()

    obs, _ = env.reset()
    for _ in range(MAX_STEPS):
        ws = torch.tensor(obs["which_side"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        c1 = torch.tensor(obs["choice1"],    dtype=torch.float32, device=DEVICE).unsqueeze(0)
        c2 = torch.tensor(obs["choice2"],    dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, _ = net({"which_side": ws, "choice1": c1, "choice2": c2})
        action = torch.argmax(logits, dim=-1).item()
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break

    return env._build_cliques()

def random_split_hla(
    parsed_hla
):
    """
    Given parsed_hla like:
      [
        [['A*02:01','A*01:01'], ['A*32:01','A*01:04']],
        [['B*44:02'],         ['B*57:01']],
        ...
      ]
    Returns two equal‐length lists (groupA, groupB) where for each locus:
      - One sublist was randomly picked for A, the other for B
      - A single random element was drawn from each sublist
    """
    groupA, groupB = [], []
    for locus_pair in parsed_hla:
        alleles1, alleles2 = locus_pair

        # randomly decide which goes to A vs B
        if random.choice([True, False]):
            a_choices, b_choices = alleles1, alleles2
        else:
            a_choices, b_choices = alleles2, alleles1

        # pick one allele from each
        groupA.append(random.choice(a_choices))
        groupB.append(random.choice(b_choices))

    return groupA, groupB

# ---- Main Execution ----

if __name__ == "__main__":
    # ---- Configuration ----
    CONFIG_PATH = "../config.json"
    MODEL_PATH = "clique_ac_model_allgraphs.pth"
    DEVICE = "cpu"
    MAX_STEPS = 50

    # load graph
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    graph_pkl = os.path.join(os.path.dirname(CONFIG_PATH), cfg['graph_path'], cfg['donors_file'] + '.pkl')
    with open(graph_pkl, 'rb') as f:
        G = pickle.load(f)

    # prepare HLA lists
    HLA_list = create_hla(CONFIG_PATH)

    best, model, random_scores = [], [], []
    for hla in tqdm(HLA_list, desc="HLA Eval"):
        best.append(generate_hla_combinations(G, hla))
        c1, c2 = impute_hla(hla)
        model.append(calculate_clique_weight(G, c1) + calculate_clique_weight(G, c2))
        rc1, rc2 = random_split_hla(hla)
        random_scores.append(calculate_clique_weight(G, rc1) + calculate_clique_weight(G, rc2))

    # save scores to a CSV file
    scores = np.vstack([best, random_scores, model]).T
    np.savetxt(
        "scores.csv",
        scores,
        fmt="%.6f",
        delimiter=",",
        header="best,random,model",
        comments=""
    )
    print("Saved scores to scores.csv")
