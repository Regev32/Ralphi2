import json
import pandas as pd
from itertools import groupby
import networkx as nx
import os
import pickle
from tqdm import tqdm


def create_allele_dict(hla_list, config):
    allele_dict = {
        f"{locus}": set()
        for locus in config['allowed_loci']
    }

    for hla in tqdm(hla_list, desc="Creating allele dictionary"):
        for locus in hla:
            for allele in locus:
                for node in allele:
                    if node.split('*')[0] == "nan":
                        pass
                    else:
                        allele_dict[f"{node.split('*')[0]}"].add(node)

    # now convert to JSON‐friendly lists
    json_ready = {
        key: sorted(vals)
        for key, vals in allele_dict.items()
    }

    json_path = '../' +  os.path.join(
        config['graph_path'],
        config['donors_file'] + '_allele.json'
    )
    with open(json_path, 'w') as nf:
        json.dump(json_ready, nf, indent=2)


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


def create_graph(hla_list):
    graph_dict = {}
    # Outer progress bar for HLA entries
    for hla in tqdm(hla_list, desc="Building graph from HLA list"):
        for i in range(len(hla) - 1):
            locus1 = hla[i]
            for j in range(i + 1, len(hla)):
                locus2 = hla[j]
                for allele1 in locus1:
                    for allele2 in locus2:
                        weight = (len(allele1) * len(allele2)) ** -1
                        for node1 in allele1:
                            for node2 in allele2:
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

    # Read only the HLA column
    donors_csv = os.path.join(cfg['donors_folder'], cfg['donors_file'] + '.csv')
    df = pd.read_csv('../' + donors_csv, usecols=[1], header=0, names=['raw_hla'])

    # Parse each HLA string
    tqdm.pandas(desc="Parsing HLA strings")
    df['parsed_hla'] = df['raw_hla'].astype(str).progress_apply(parse_hla_nested)
    hla_list = df['parsed_hla'].tolist()
    create_allele_dict(hla_list, cfg)
    # Build graph and save
    G = create_graph(hla_list)

    # Save graph object
    out_file = '../' +  os.path.join(cfg['graph_path'], cfg['donors_file'] + '.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {out_file}")

if __name__ == "__main__":
    main()