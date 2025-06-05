import json
import ast
import pandas as pd
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    # parse booleans
    cfg['first_row_headers'] = cfg.get('first_row_headers', 'True') == 'True'
    cfg['allowed_loci']      = ast.literal_eval(cfg.get('allowed_loci', "set()"))
    return cfg

def filter_hla_loci(hla_string, allowed_loci):
    parts = hla_string.split('^')
    return '^'.join(p for p in parts if p.split('*',1)[0] in allowed_loci)

def main():
    # --- load config ---
    config     = load_config('../../config.json')
    donors_csv = os.path.join(config['donors_folder'], config['donors_file'] + '.csv')
    allowed    = config['allowed_loci']
    hdr        = 0 if config['first_row_headers'] else None

    # --- read CSV ---
    df = pd.read_csv(donors_csv, header=hdr)
    # if header=None, pandas will assign integer column names
    # we assume HLA is in the second column regardless
    hla_col = df.columns[1]

    # --- filter ---
    df[hla_col] = df[hla_col].astype(str).apply(lambda x: filter_hla_loci(x, allowed))

    # --- save back ---
    df.to_csv(donors_csv, index=False, header=config['first_row_headers'])
    print(f"Filtered to only loci {allowed}")

if __name__ == "__main__":
    main()