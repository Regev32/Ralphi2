import json
import pandas as pd
import os
from tqdm import tqdm
from graph_creation import parse_hla_nested


def main(config_path="../config.json"):
    # Load configuration
    with open(config_path) as f:
        cfg = json.load(f)
    os.makedirs('../' + cfg['graph_path'], exist_ok=True)

    # Read only the HLA column
    donors_csv = os.path.join('..', cfg['donors_folder'], cfg['donors_file'] + '.csv')
    df = pd.read_csv(donors_csv, usecols=[1], header=0, names=['raw_hla'])

    # Parse each HLA string
    tqdm.pandas(desc="Parsing HLA strings")
    df['parsed_hla'] = df['raw_hla'].astype(str).progress_apply(parse_hla_nested)
    hla_list = df['parsed_hla'].tolist()

    with open(f"../data/typing/length10{cfg["donors_file"]}.txt", 'w') as f:
        for hla in hla_list:
            length = 0
            for locus in hla:
                for chromosome in locus:
                    for _ in chromosome:
                        length += 1
            if length < 50:
                    f.write(str(hla) + '\n')
    f.close()

if __name__ == "__main__":
    main()
