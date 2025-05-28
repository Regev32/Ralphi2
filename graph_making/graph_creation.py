import os
import json
import pandas as pd
from itertools import groupby
import networkx as nx
import pickle
from tqdm import tqdm

# 1) Determine your project root (one level up from this script)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# 2) Load config.json from project root
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
with open(CONFIG_PATH, 'r') as f:
    cfg = json.load(f)

# 3) Build these once
DONORS_CSV = os.path.join(
    BASE_DIR,
    cfg['donors_folder'],
    cfg['donors_file'] + '.csv'
)
GRAPH_DIR    = os.path.join(BASE_DIR, cfg['graph_path'])
OUTPUT_PKL   = os.path.join(GRAPH_DIR, cfg['donors_file'] + '.pkl')
LOCI_OUTPUT  = os.path.join(GRAPH_DIR, cfg['donors_file'] + '_loci.json')
# Ensure output directory exists
os.makedirs(GRAPH_DIR, exist_ok=True)


def parse_hla_nested(hla_string):
    """
    Parse a raw HLA string into a nested list by locus.
    Example input: "A*02:01+A*02:02^B*07:05+B*14:02^…"
    """
    # Split into locus parts on '^', then split each on '+'
    alleles = [
        allele.strip()
        for locus_part in hla_string.split('^')
        for allele in locus_part.split('+')
        if allele.strip()
    ]
    # Group by the part before the '*' so that we keep each locus together
    return [
        list(group)
        for _, group in groupby(alleles, key=lambda a: a.split('*', 1)[0])
    ]


def create_allele_dict(hla_list):
    """
    From a list of parsed HLAs (list of lists of allele-groups),
    build a dict mapping locus -> unique set of allele-strings.
    Then save to LOCI_OUTPUT.
    """
    allele_dict = {locus: set() for locus in cfg['allowed_loci']}

    for entry in tqdm(hla_list, desc="Collecting alleles by locus"):
        for locus_group in entry:
            # locus name is first two chars before '*' (e.g. 'A')
            locus_name = locus_group[0].split('*', 1)[0]
            if locus_name in allele_dict:
                allele_dict[locus_name].update(locus_group)

    # Convert to sorted lists and dump JSON
    allele_json = {locus: sorted(list(vals)) for locus, vals in allele_dict.items()}
    with open(LOCI_OUTPUT, 'w') as f:
        json.dump(allele_json, f, indent=2)


def create_graph(hla_list):
    """
    Build the weighted co-occurrence graph:
    For each HLA entry (list of locus-groups),
    connect every allele in locus i to every allele in locus j
    with weight = 1/(len(allele_i)*len(allele_j)), then sum across the dataset.
    """
    edge_weights = {}

    for entry in tqdm(hla_list, desc="Accumulating edge weights"):
        # flatten entry into per-locus lists
        for i in range(len(entry) - 1):
            for j in range(i + 1, len(entry)):
                group_i = entry[i]
                group_j = entry[j]
                inv_denom = 1.0 / (len(group_i) * len(group_j))
                for allele_i in group_i:
                    for allele_j in group_j:
                        key = tuple(sorted((allele_i, allele_j)))
                        edge_weights[key] = edge_weights.get(key, 0.0) + inv_denom

    G = nx.Graph()
    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)
    return G


def main():
    # 1) Read the CSV
    if not os.path.isfile(DONORS_CSV):
        raise FileNotFoundError(f"Could not find HLA CSV at {DONORS_CSV}")
    df = pd.read_csv(DONORS_CSV, usecols=[1], header=0, names=['raw_hla'])

    # 2) Parse HLA strings
    tqdm.pandas(desc="Parsing HLA strings")
    df['parsed_hla'] = df['raw_hla'].astype(str).progress_apply(parse_hla_nested)
    hla_list = df['parsed_hla'].tolist()

    # 3) Build allele-dict JSON
    create_allele_dict(hla_list)

    # 4) Build the full co-occurrence graph and pickle it
    G = create_graph(hla_list)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved full HLA graph to {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
