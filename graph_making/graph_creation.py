import json
import ast
import networkx as nx
import os
import pickle
from tqdm import tqdm
from itertools import groupby


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


def create_allele_dict(hla_list, config):
    allele_dict = {f"{locus}": {}for locus in config['allowed_loci']}

    for hla in tqdm(hla_list, desc="Creating allele dictionary"):
        for locus in hla:
            for allele in locus:
                for node in allele:
                    if node.split('*')[0] == "nan":
                        pass
                    else:
                        if node in allele_dict[f"{node.split('*')[0]}"]:
                            allele_dict[f"{node.split('*')[0]}"][node] += 1
                        else:
                            allele_dict[f"{node.split('*')[0]}"][node] = 1

    return allele_dict


def create_graph(hla_list):
    graph_dict = {}
    # Outer progress bar for HLA entries
    for hla in tqdm(hla_list, desc="Building graph from HLA list"):
        for i in range(len(hla) - 1):
            locus1 = hla[i]
            for j in range(i + 1, len(hla)):
                locus2 = hla[j]
                for chromosome1 in locus1:
                    for chromosome2 in locus2:
                        weight = (len(chromosome1) * len(chromosome2))
                        if weight > 50:
                            continue
                        weight = weight ** -1
                        for node1 in chromosome1:
                            for node2 in chromosome2:
                                edge = (node1, node2)
                                graph_dict[edge] = graph_dict.get(edge, 0) + weight
    G = nx.Graph()
    G.add_weighted_edges_from((u, v, w) for (u, v), w in graph_dict.items())
    return G


def main(config_path="../config.json"):
    # Load configuration
    with open(config_path) as f:
        cfg = json.load(f)
    os.makedirs('../' + cfg['graph_path'], exist_ok=True)

    path = f"../{cfg['donors_folder']}/length10{cfg['donors_file']}.txt"
    hla_list = []

    with open(path, 'r') as f:
        for line in f:
            nested_hla = ast.literal_eval(line.strip())
            hla_list.append(nested_hla)

    json_ready = create_allele_dict(hla_list, cfg)
    G = create_graph(hla_list)

    # Save
    path = os.path.join('..', cfg['graph_path'], cfg['donors_file'])

    json_path = path + '_allele.json'
    with open(json_path, 'w') as nf:
        json.dump(json_ready, nf, indent=2)
    print(f"Allele dictionary saved to {json_path}")

    out_file = path + '.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {out_file}")


    i = 0
    tokenizer = {}
    for key in json_ready.keys():
        for value in json_ready[key]:
            tokenizer[value] = i
            i += 1

    with open("../graphs/indexes.json", "w") as f:
        json.dump(tokenizer, f, indent=2)
if __name__ == "__main__":
    main()