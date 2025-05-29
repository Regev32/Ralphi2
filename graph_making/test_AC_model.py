#!/usr/bin/env python3
import os
import json
import pickle
import random
import itertools
import csv

import networkx as nx
import torch

from graph_actor_critic import GraphPartitionEnv, ActorCritic


def random_hla_string(alleles_by_locus, loci):
    parts = []
    for locus in loci:
        grp = []
        for _ in range(2):
            k = random.choice([1, 2, 3])
            picks = random.sample(alleles_by_locus[locus], k)
            grp.append("/".join(picks))
        parts.append("+".join(grp))
    return "^".join(parts)


def make_subgraph_for_hla(G_full, hla_string):
    groups = [
        chunk.strip()
        for locus_part in hla_string.split('^')
        for chunk in locus_part.split('+')
        if chunk.strip()
    ]
    var2grp = {allele: grp for grp in groups for allele in grp.split('/')}
    valid = [
        grp for grp in groups
        if any(allele in G_full for allele in grp.split('/'))
    ]

    subG = nx.Graph()
    subG.add_nodes_from(valid)

    for grp in valid:
        for allele in grp.split('/'):
            if allele not in G_full:
                continue
            for nbr, data in G_full[allele].items():
                if nbr in var2grp:
                    tgt = var2grp[nbr]
                    if tgt == grp:
                        continue
                    w = data.get('weight', 1.0)
                    if subG.has_edge(grp, tgt):
                        subG[grp][tgt]['weight'] += w
                    else:
                        subG.add_edge(grp, tgt, weight=w)
    return subG


def calculate_clique_weight(clique_nodes, subG):
    total = 0.0
    for u, v in itertools.combinations(clique_nodes, 2):
        if subG.has_edge(u, v):
            total += subG[u][v]['weight']
    return total


def brute_force_best(subG):
    nodes = list(subG.nodes())
    best = -float('inf')
    for assign in itertools.product([1, -1], repeat=len(nodes)):
        clique1 = [nodes[i] for i, a in enumerate(assign) if a == 1]
        clique2 = [nodes[i] for i, a in enumerate(assign) if a == -1]
        w1 = calculate_clique_weight(clique1, subG)
        w2 = calculate_clique_weight(clique2, subG)
        best = max(best, w1 + w2)
    return best


def eval_ac_greedy(hla_string, model, G_full, cfg):
    env = GraphPartitionEnv(hla_string, cfg)
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    done = False
    while not done:
        st = torch.from_numpy(state).float()
        probs, _ = model(st)
        action = torch.argmax(probs).item()
        out = env.step(action)
        state, _, done, _ = out

    clique1, clique2 = env.cliques
    subG = make_subgraph_for_hla(G_full, hla_string)
    return calculate_clique_weight(clique1, subG) + calculate_clique_weight(clique2, subG)


def eval_ac_best_of_n(hla_string, model, G_full, cfg, n_trials=20):
    best = -float('inf')
    for _ in range(n_trials):
        score = eval_ac_greedy(hla_string, model, G_full, cfg)
        best = max(best, score)
    return best


def main():
    # --- Load config & paths ---
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    with open(os.path.join(BASE_DIR, 'config.json')) as f:
        cfg = json.load(f)

    graph_dir = os.path.join(BASE_DIR, cfg['graph_path'])
    full_pkl = os.path.join(graph_dir, cfg['donors_file'] + '.pkl')
    alleles_j = os.path.join(graph_dir, cfg['donors_file'] + '_loci.json')
    model_pth = os.path.join(BASE_DIR, cfg['save_model'])
    donors_csv = os.path.join(BASE_DIR, cfg['donors_folder'], cfg['donors_file'] + '.csv')

    N_REAL = cfg.get('n_real_hlas', 5)
    N_TRIALS = cfg.get('n_trials', 20)
    n_sub = cfg.get('n_subgraphs', 10)

    # --- Load data ---
    G_full = pickle.load(open(full_pkl, 'rb'))
    with open(alleles_j) as f:
        alleles_by_locus = json.load(f)
    loci = list(alleles_by_locus.keys())

    # --- Select real HLAs ---
    real_hlas = []
    with open(donors_csv) as f:
        if cfg.get('first_row_headers', False):
            next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 1 and parts[1].strip():
                real_hlas.append(parts[1].strip())
    real_hlas = random.sample(real_hlas, min(N_REAL, len(real_hlas)))

    # --- Load model ---
    dummy = GraphPartitionEnv(random_hla_string(alleles_by_locus, loci), cfg)
    model = ActorCritic(
        state_dim=dummy.observation_space.shape[0],
        action_dim=dummy.action_space.n
    )
    model.load_state_dict(torch.load(model_pth, map_location='cpu'))
    model.eval()

    # --- Run tests & collect results ---
    results = []
    for hla_string in real_hlas:
        subG = make_subgraph_for_hla(G_full, hla_string)
        brute = brute_force_best(subG)
        ac_scr = eval_ac_best_of_n(hla_string, model, G_full, cfg, N_TRIALS)
        results.append((hla_string, brute, ac_scr))
        print(f"REAL          | brute={brute:.3f}, AC={ac_scr:.3f}")

    for i in range(1, n_sub + 1):
        hla_string = random_hla_string(alleles_by_locus, loci)
        subG = make_subgraph_for_hla(G_full, hla_string)
        brute = brute_force_best(subG)
        ac_scr = eval_ac_best_of_n(hla_string, model, G_full, cfg, N_TRIALS)
        results.append((hla_string, brute, ac_scr))
        print(f"RANDOM [{i}/{n_sub}] | brute={brute:.3f}, AC={ac_scr:.3f}")

    # --- Save results ---
    out_csv = os.path.join(graph_dir, 'test_results.csv')
    with open(out_csv, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['HLA', 'score_with_brute_force', 'score_with_actor_critic'])
        writer.writerows(results)
    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    main()
