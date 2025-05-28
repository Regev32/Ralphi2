import json
import os
import pickle
from pyvis.network import Network
import webbrowser


def plot_hla_graph(G, output_html):
    """
    Render a NetworkX graph G using PyVis and open the HTML file in the browser.

    Parameters
    ----------
    G : networkx.Graph
        The graph to visualize.
    output_html : str
        Path (including filename) for the output HTML file.
    """
    net = Network(height='800px', width='100%', notebook=False)
    net.barnes_hut()
    net.from_nx(G)
    net.save_graph(output_html)
    webbrowser.open(f'file://{os.path.abspath(output_html)}')


def main(config_path="../config.json"):
    # Load configuration
    with open(config_path) as f:
        cfg = json.load(f)

    # Load the pickled graph
    graph_file = os.path.join(cfg['graph_path'], cfg['donors_file'] + '.pkl')
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    # Plot to HTML
    output_html = os.path.join(cfg['graph_path'], cfg['output_html'])
    plot_hla_graph(G, output_html)
    print(f"Graph HTML saved and opened at {output_html}")


if __name__ == '__main__':
    main()
