import networkx as nx
import os
import pandas as pd
import json
import random
import pickle as pkl


def parse_hla_nested(hla_string):
    """
    Parse a raw HLA string into a nested list by locus.
    """
    return [
        list(group)
        for locus, group in groupby(
            [list(set(allele.split('/')))
             for locus_part in hla_string.split('^')
             for allele in locus_part.split('+')],
            key=lambda x: x[0].split('*', 1)[0]
        )
    ]


def extract_hla(conf_path):
    with open(conf_path, 'r') as f:
        cfg = json.load(f)

    donors_csv = f"../{cfg['donors_folder']}/{cfg['donors_file']}.csv"
    with open(donors_csv, 'r') as f:
        lines = f.readlines()
    n_hla = cfg['n_subgraphs']
    hla_list = lines[1:n_hla + 1]
    hla_list = [line.strip().split(',')[1] for line in hla_list]
    return hla_list

def create_hla(conf_path):
    with open(conf_path, 'r') as f:
        cfg = json.load(f)

    loci_dict_path = f"../{cfg['graph_path']}/{cfg['donors_file']}_allele.json"
    with open(loci_dict_path, 'r') as f:
        loci_dict = json.load(f)

    n_hla = cfg['n_subgraphs']
    hla_list = []
    for i in range(n_hla):
        hla = []
        for key in loci_dict.keys():
            k = random.randint(1, 3)
            l = random.randint(1, 3)

            sample = random.sample(loci_dict[key], l + k)
            mom_chromosome = '/'.join(sample[0:l])
            dad_chromosome = '/'.join(sample[l:l + k])

            locus = '+'.join([mom_chromosome, dad_chromosome])
            hla.append(locus)
        hla = '^'.join(hla)
        hla_list.append(hla)

    return hla_list

def calculate_clique_weight(G, alleles):
    return sum([G[u][v]['weight'] for i, u in enumerate(alleles) for j, v in enumerate(alleles) if i < j and G.has_edge(u, v)])

def create_subgraph(conf_path, hla):
    with open(conf_path, 'r') as f:
        cfg = json.load(f)
    hla = parse_hla_nested(hla)

    graph_path = f"../{cfg['graph_path']}/{cfg['donors_file']}.pkl"
    with open(graph_path, 'rb') as f:
        graph = pkl.load(f)






if __name__ == "__main__":
    conf_path = "../config.json"
    real_list = extract_hla(conf_path)
    for hla in real_list:
        print(hla)
    print('_' * 50)
    fake_list = create_hla(conf_path)
    for hla in fake_list:
        print(hla)

