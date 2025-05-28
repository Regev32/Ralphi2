#!/usr/bin/env python3
import os
import json
import pickle
import random
import itertools
import csv
import statistics

import networkx as nx
import numpy as np
import torch

from graph_actor_critic import GraphPartitionEnv, ActorCritic

# --- Load config & paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
with open(os.path.join(BASE_DIR, 'config.json')) as f:
    cfg = json.load(f)

n_sub     = cfg.get('n_subgraphs', 10)
graph_dir = os.path.join(BASE_DIR, cfg['graph_path'])
full_pkl  = os.path.join(graph_dir, cfg['donors_file'] + '.pkl')
alleles_j = os.path.join(graph_dir, cfg['donors_file'] + '_loci.json')
model_pth = os.path.join(BASE_DIR, cfg['save_model'])
donors_csv= os.path.join(BASE_DIR,
                         cfg['donors_folder'],
                         cfg['donors_file'] + '.csv')

# Number of greedy rollouts per HLA to take max over
N_TRIALS = 20
# Number of real HLAs to include
N_REAL   = 5

# --- Load data ---
G_full = pickle.load(open(full_pkl, 'rb'))
with open(alleles_j) as f:
    alleles_by_locus = json.load(f)
loci = list(alleles_by_locus.keys())

# --- Pull real HLAs from your CSV file ---
real_hlas = []
with open(donors_csv) as f:
    if cfg.get('first_row_headers', False):
        next(f)
    for line in f:
        parts = line.strip().split(',')
        if len(parts) > 1 and parts[1].strip():
            real_hlas.append(parts[1].strip())
# Sample up to N_REAL unique entries
real_hlas = random.sample(real_hlas, min(N_REAL, len(real_hlas)))

# --- Helpers ---
def random_hla_string():
    parts = []
    for locus in loci:
        grp = []
        for _ in range(2):
            k = random.choice([1,2,3])
            picks = random.sample(alleles_by_locus[locus], k)
            grp.append("/".join(picks))
        parts.append("+".join(grp))
    return "^".join(parts)

def make_subgraph_for_hla(G_full, hla_string):
    groups = [
        chunk.strip()
        for locus_part in hla_string.split('^')
        for chunk     in locus_part.split('+')
        if chunk.strip()
    ]
    var2grp = {v: grp for grp in groups for v in grp.split('/')}

    valid = [grp for grp in groups
             if any(v in G_full for v in grp.split('/'))]

    sub = nx.Graph()
    sub.add_nodes_from(valid)
    for grp in valid:
        for v in grp.split('/'):
            if v not in G_full: continue
            for nbr, data in G_full[v].items():
                if nbr in var2grp:
                    tgt = var2grp[nbr]
                    if tgt == grp: continue
                    w = data.get('weight', 1.0)
                    if sub.has_edge(grp, tgt):
                        sub[grp][tgt]['weight'] += w
                    else:
                        sub.add_edge(grp, tgt, weight=w)
    return sub

def brute_force_best(subG):
    nodes = list(subG.nodes())
    best  = -np.inf
    for assign in itertools.product([1,-1], repeat=len(nodes)):
        w = 0.0
        for sign in (1, -1):
            clique = [nodes[i] for i,a in enumerate(assign) if a==sign]
            for u,v in itertools.combinations(clique, 2):
                if subG.has_edge(u,v):
                    w += subG[u][v]['weight']
        best = max(best, w)
    return best

def eval_ac_greedy(hla_string, model):
    env = GraphPartitionEnv(hla_string, cfg)
    state = env.reset()
    done  = False
    while not done:
        st     = torch.from_numpy(state).float()
        probs, _ = model(st)
        action = torch.argmax(probs).item()
        state, _, done, _ = env.step(action)
    return env._calc_weight()

def eval_ac_best_of_n(hla, model, n=N_TRIALS):
    best = -float('inf')
    for _ in range(n):
        score = eval_ac_greedy(hla, model)
        best  = max(best, score)
    return best

# --- Load trained AC model ---
dummy = GraphPartitionEnv(random_hla_string(), cfg)
model = ActorCritic(
    state_dim = dummy.observation_space.shape[0],
    action_dim= dummy.action_space.n
)
model.load_state_dict(torch.load(model_pth, map_location='cpu'))
model.eval()

# --- Run tests & collect results ---
results = []

# First, the real HLAs
for hla in real_hlas:
    subG   = make_subgraph_for_hla(G_full, hla)
    brute  = brute_force_best(subG)
    ac_scr = eval_ac_best_of_n(hla, model)
    results.append((hla, brute, ac_scr))
    print(f"REAL    | brute={brute:.3f}, AC(best of {N_TRIALS})={ac_scr:.3f}")

# Then, n_sub random HLAs
for i in range(1, n_sub+1):
    hla    = random_hla_string()
    subG   = make_subgraph_for_hla(G_full, hla)
    brute  = brute_force_best(subG)
    ac_scr = eval_ac_best_of_n(hla, model)
    results.append((hla, brute, ac_scr))
    print(f"[{i}/{n_sub}] RANDOM | brute={brute:.3f}, AC(best of {N_TRIALS})={ac_scr:.3f}")

# --- Write CSV next to your loci JSON (in graphs/) ---
out_csv = os.path.join(graph_dir, 'test_results.csv')
with open(out_csv, 'w', newline='') as cf:
    writer = csv.writer(cf)
    writer.writerow(['HLA',
                     'score_with_brute_force',
                     'score_with_actor_critic'])
    writer.writerows(results)

print(f"\nSaved results to {out_csv}")
