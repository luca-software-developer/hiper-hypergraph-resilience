from typing import Dict, List, Tuple
import numpy as np


class Hypernetwork:
    def __init__(self, n: int, m: int, nodes_map: Dict[int, int], hyperedges_list: List[List[int]]):
        """
        Builds a hypernetwork with `n` nodes and `m` hyperedges.
        `nodes_map` stores the association between original node index and matrix one.
        `hyperedges_list` is the list of hyperedges.
        """

        # Order and size of the hypernetwork
        self.n = n
        self.m = m

        # nodes_map: original index -> matrix index
        # index_map: matrix index -> original index
        self.nodes_map = nodes_map
        self.index_map = {v: k for (k, v) in self.nodes_map.items()}
        self.hyperedges_list = hyperedges_list
        self.incidence_matrix = np.zeros(shape=(n, m))

        # We build the incidence matrix I
        for (j, hyperedge) in enumerate(hyperedges_list):
            for node in hyperedge:
                self.incidence_matrix[self.nodes_map[node], j] = 1

        # Adjacency matrix A = II^T-D and we binarize it
        # A[i,j] = 1 iff node i and node j share at least one common hyperedge
        self.adjacency_matrix = self.incidence_matrix.dot(self.incidence_matrix.T)
        np.fill_diagonal(self.adjacency_matrix, 0)
        self.adjacency_matrix = np.where(self.adjacency_matrix > 1, 1, self.adjacency_matrix)

    def order(self) -> int:
        """
        Returns the order of the hypernetwork, that is the number of its nodes.
        """
        return self.n

    def size(self) -> int:
        """
        Returns the size of the hypernetwork, that is the number of its hyperedges.
        """
        return self.m

    def get_hyperedges(self, node: int) -> List[int]:
        """
        Returns the hyperedges index containing the node.
        :param node:
        """
        return np.where(self.incidence_matrix[self.nodes_map[node], :] == 1)[0]

    def get_nodes(self, hyperedge: int) -> List[int]:
        """
        Returns the nodes contained in the hyperedge.
        :param hyperedge:
        """
        return self.hyperedges_list[hyperedge]

    def get_neighbors(self, node: int) -> List[int]:
        """
        Returns the neighbours of the node.
        """
        return list(filter(lambda nn: self.nodes_map[node] != nn,
                           map(lambda n: self.index_map[n],
                               np.where(self.adjacency_matrix[self.nodes_map[node], :] >= 1)[0])))

    def deg(self, node: int) -> int:
        """
        Returns the degree of a node. The degree is defined as the number of neighbours of the node.
        """
        return self.adjacency_matrix[self.nodes_map[node], :].sum()

    def hyperdeg(self, node: int) -> int:
        """
        Returns the hyperdegree of a node. The hyperdegree is defined as the number of
        hyperedges containing the node.
        """
        return self.incidence_matrix[self.nodes_map[node], :].sum()

    def avg_deg(self) -> float:
        """
        Returns the average degree of the hypernetwork.
        """
        return np.sum([self.deg(node) for node in self.nodes_map.keys()]) / self.n

    def avg_hyperdegree(self) -> float:
        """
        Returns the average hyperdegree of the hypernetwork.
        """
        return np.sum([self.hyperdeg(node) for node in self.nodes_map.keys()]) / self.n

    def hyperedge_size(self, hyperedge: int) -> int:
        """
        Returns the cardinality of the given hyperedge, i.e., the number of nodes it contains.
        :param hyperedge:
        """
        return len(self.hyperedges_list[hyperedge])

    def avg_hyperedge_size(self) -> float:
        """
        Returns the average hyperedge size of the hypernetwork.
        """
        return np.sum([self.hyperedge_size(hyperedge) for hyperedge in range(self.m)]) / self.m

    def line_graph(self) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Constructs and returns the line graph of the hypernetwork.
        """
        # Initialize the line graph structure
        line_graph_nodes = list(range(len(self.hyperedges_list)))
        line_graph_edges = list()

        # Compare each pair of edges in the hypergraph
        for i in range(len(self.hyperedges_list)):
            for j in range(i + 1, len(self.hyperedges_list)):
                if set(self.hyperedges_list[i]).intersection(self.hyperedges_list[j]):
                    line_graph_edges.append((i, j))

        return line_graph_nodes, line_graph_edges

    def print_info(self):
        """
        Prints information about the hypernetwork.
        """
        print(f'Order: {self.order()}')
        print(f'Size: {self.size()}')
        print(f'Avg degree: {self.avg_deg():.2f}')
        print(f'Avg hyperdegree: {self.avg_hyperdegree():.2f}')
        print(f'Avg hyperedge size: {self.avg_hyperedge_size():.2f}')
