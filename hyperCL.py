import random

import numpy as np


def generate_powerlaw_degrees(num_nodes, theta=2.0, d_min=1, d_max=20):
    """
    Generate a degree sequence following a power-law distribution p(d) ~ d^(-theta).

    :param num_nodes: The number of nodes (length of the degree sequence).
    :param theta: The exponent of the power-law distribution (default is 2).
    :param d_min: The minimum degree value (default is 1).
    :param d_max: The maximum degree value (if None, it is set to a large number).
    :return: A list of node degrees.
    """
    if d_max is None:
        d_max = num_nodes * 10  # Set a reasonable upper bound if not provided

    # Inverse transform sampling to generate a degree sequence
    random_numbers = np.random.uniform(0, 1, num_nodes)
    degrees = ((d_max ** (1 - theta) - d_min ** (1 - theta)) * random_numbers + d_min ** (1 - theta)) ** (
                1 / (1 - theta))
    return degrees.astype(int)  # Return the degrees as integers


def generate_uniform_hyperedge_sizes(num_hyperedges, min_size=1, max_size=10):
    """
    Generate a hyperedge size sequence using a uniform distribution.

    :param num_hyperedges: The number of hyperedges.
    :param min_size: The minimum size of a hyperedge (default is 1).
    :param max_size: The maximum size of a hyperedge (default is 10).
    :return: A list of hyperedge sizes.
    """
    # Generate random integers for hyperedge sizes
    hyperedge_sizes = np.random.randint(min_size, max_size + 1, size=num_hyperedges)
    return hyperedge_sizes


def hypercl(hyperedge_sizes, node_degrees):
    """
    Generate a random hypergraph using the HyperCL algorithm.

    :param hyperedge_sizes: List of sizes of hyperedges.
    :param node_degrees: List of degrees for each node.
    :return: A random hypergraph represented as a list of hyperedges (each hyperedge is a set of nodes).
    """

    # Number of nodes and hyperedges
    num_nodes = len(node_degrees)
    num_hyperedges = len(hyperedge_sizes)

    # Initialize the hypergraph (empty set of hyperedges)
    hypergraph = []

    # Generate hyperedges
    for i in range(num_hyperedges):
        hyperedge_size = hyperedge_sizes[i]
        hyperedge = set()

        # Sample nodes with probability proportional to their degree
        while len(hyperedge) < hyperedge_size:
            selected_node = random.choices(range(num_nodes), weights=node_degrees)[0]
            hyperedge.add(selected_node)  # Add the selected node (ignore duplicates)

        hypergraph.append(hyperedge)

    return hypergraph


if __name__ == '__main__':
    num_nodes = 1000
    num_hyperedges = 1000
    d_min, d_max = 1, 20
    min_size, max_size = 1, 10
    for theta in (2.0, 2.1, 2.3, 2.5):
        degree_sequence = generate_powerlaw_degrees(num_nodes, theta, d_min, d_max)
        hyperedge_sizes = generate_uniform_hyperedge_sizes(num_hyperedges, min_size, max_size)
        hypergraph = hypercl(hyperedge_sizes, degree_sequence)

        with open(f'data/hypercl_{int(theta * 10)}.txt', 'w') as outfile:
            hyperedges = (' '.join(map(str, sorted(he))) for he in hypergraph)
            outfile.write('\n'.join(hyperedges))
