import json
import os
import pickle
import networkx as nx

# ------------------
# Configuration
# ------------------
CONFIG_PATH = "../config.json"
HLA_STRING = (
    "A*01:01+A*02:01/A*02:02^"
    "B*07:33+B*14:02^"
    "C*06:02+C*08:02^"
    "DRB1*01:04+DRB1*01:01^"
    "DQB1*02:01+DQB1*05:01"
)

def parse_groups(hla_string):
    """
    Split on '^' then '+', keep '/' inside each chunk.
    Returns a list of allele-group strings, e.g. 'A*02:01/A*02:02'.
    """
    groups = []
    for locus_part in hla_string.split('^'):
        for chunk in locus_part.split('+'):
            g = chunk.strip()
            if g:
                groups.append(g)
    return groups


def build_variant_map(groups):
    """
    From e.g. ['A*02:01/A*02:02', 'A*01:01', ...]
    return:
      - group_nodes: the original list of group-strings
      - var2group: dict mapping each individual variant -> group-string
    """
    var2group = {}
    for grp in groups:
        for var in grp.split('/'):
            var2group[var] = grp
    return groups, var2group


def make_subgraph_for_hla(G_full, hla_string):
    """
    Build an unweighted subgraph for the given HLA string.

    Parameters
    ----------
    G_full : networkx.Graph
        The full HLA variant graph.
    hla_string : str
        HLA groups encoded as on input, e.g. 'A*01:01+A*02:01/A*02:02^...'.

    Returns
    -------
    networkx.Graph
        Subgraph with group-nodes and inherited edges.
    """
    # Parse patient groups
    groups = parse_groups(hla_string)
    group_nodes, var2group = build_variant_map(groups)

    # Warn if any variants missing entirely
    missing = [v for v in var2group if v not in G_full]
    if missing:
        print("Warning: these variants not in full graph and are skipped:", missing)

    # Only keep groups that have at least one variant in G_full
    valid_groups = {
        grp for grp in group_nodes
        if any(var in G_full for var in grp.split('/'))
    }

    # Build unweighted subgraph
    subG = nx.Graph()
    subG.add_nodes_from(valid_groups)

    # For each group, inherit edges from each variant
    for grp in valid_groups:
        for var in grp.split('/'):
            if var not in G_full:
                continue
            for nbr in G_full[var]:
                if nbr in var2group:
                    target_grp = var2group[nbr]
                    if target_grp != grp:
                        subG.add_edge(grp, target_grp)

    return subG


def main():
    # Load config & full graph
    cfg = json.load(open(CONFIG_PATH))
    base = cfg['graph_path']
    name = cfg['donors_file']
    full_pkl = os.path.join(base, f"{name}.pkl")
    print(f"Loading full graph from {full_pkl}")
    G_full = pickle.load(open(full_pkl, 'rb'))

    # Build subgraph for the given HLA string
    subG = make_subgraph_for_hla(G_full, HLA_STRING)

    # Save
    out_dir = os.path.join(base, "subgraph")
    os.makedirs(out_dir, exist_ok=True)
    out_pkl = os.path.join(out_dir, "patient_subgraph.pkl")
    with open(out_pkl, 'wb') as f:
        pickle.dump(subG, f)

    print(f"Subgraph has {subG.number_of_nodes()} nodes and "
          f"{subG.number_of_edges()} edges -> {out_pkl}")


if __name__ == "__main__":
    main()
