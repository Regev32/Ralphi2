"""
This is the main file for the project, now set up for profiling.
"""
import json
import cProfile
import pstats
from train_clique_on_file import load_all_graphs_from_file, train_actor_critic_on_file


if __name__ == '__main__':
    with cProfile.Profile() as pr:
        conf_path = "../config.json"
        with open(conf_path) as json_file:
            conf = json.load(json_file)
        donors_path = conf["donors_file"]
        txt_path = f"../data/typing/length10{donors_path}.txt" # Adjust path as necessary
        graphs = load_all_graphs_from_file(txt_path)
        print(f"Loaded {len(graphs)} graphs from {txt_path}")

        trained_net = train_actor_critic_on_file(
            graph_list=graphs,
            num_epochs=3,
            gamma=0.99,
            lr=1e-4,
            print_every=20,
            max_steps_per_env=30,
        )

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
    stats.dump_stats('stats_clique_training.prof') # Changed output file name for clarity