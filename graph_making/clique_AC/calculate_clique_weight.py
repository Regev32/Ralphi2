import json
import os
import pickle as pkl


def calculate_clique_weight( alleles = None, conf_path = "../../config.json"):
    with open(conf_path) as json_file:
        CONF = json.load(json_file)

    graph_path = os.path.join("..", "..", CONF['graph_path'], CONF['donors_file'] + '.pkl')
    with open(graph_path, 'rb') as f:
        G = pkl.load(f)
    return sum(
        G[u][v]['weight']
        for i, u in enumerate(alleles)
        for j, v in enumerate(alleles)
        if i < j and G.has_edge(u, v)
    )
