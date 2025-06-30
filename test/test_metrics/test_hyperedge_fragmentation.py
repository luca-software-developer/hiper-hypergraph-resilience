# -*- coding: utf-8 -*-
"""
test_hyperedge_fragmentation.py

Unit tests for the HyperedgeFragmentationIndex metric implementation.
Tests cover fragmentation computation, detailed analysis, and fragmentation
mapping for various perturbation scenarios.
"""

import unittest

from hiper.core.hypernetwork import Hypernetwork
from hiper.metrics.hyperedge_fragmentation import HyperedgeFragmentationIndex


class TestHyperedgeFragmentationIndex(unittest.TestCase):
    """
    Tests HyperedgeFragmentationIndex metric computation and analysis.
    Methods tested include fragmentation computation, detailed analysis,
    and individual hyperedge fragmentation mapping.
    """

    def setUp(self):
        """Create test fixtures with various hypergraph configurations."""
        self.fragmentation_metric = HyperedgeFragmentationIndex()

        # Empty hypergraph
        self.hn_empty = Hypernetwork()

        # Original hypergraph
        self.hn_original = Hypernetwork()
        self.hn_original.add_hyperedge(0, [1, 2, 3])
        self.hn_original.add_hyperedge(1, [2, 3, 4])
        self.hn_original.add_hyperedge(2, [4, 5, 6])
        self.hn_original.add_hyperedge(3, [7, 8])

        # Intact hypergraph (all nodes preserved)
        self.hn_intact = Hypernetwork()
        self.hn_intact.add_node(1)
        self.hn_intact.add_node(2)
        self.hn_intact.add_node(3)
        self.hn_intact.add_node(4)
        self.hn_intact.add_node(5)
        self.hn_intact.add_node(6)
        self.hn_intact.add_node(7)
        self.hn_intact.add_node(8)

        # Partially fragmented (some nodes removed)
        self.hn_partial = Hypernetwork()
        self.hn_partial.add_node(1)
        self.hn_partial.add_node(2)
        self.hn_partial.add_node(4)
        self.hn_partial.add_node(5)
        self.hn_partial.add_node(7)

        # Completely fragmented (all nodes removed)
        self.hn_complete = Hypernetwork()

        # Single node remaining
        self.hn_single = Hypernetwork()
        self.hn_single.add_node(3)

    def test_metric_initialization(self):
        """Test proper initialization of HyperedgeFragmentationIndex metric."""
        self.assertEqual(self.fragmentation_metric.name,
                         "Hyperedge Fragmentation Index")
        self.assertEqual(self.fragmentation_metric.symbol, "HFI")

    def test_fragmentation_empty_original(self):
        """Test fragmentation computation with empty original hypergraph."""
        hfi = HyperedgeFragmentationIndex.compute(self.hn_empty, self.hn_empty)
        self.assertEqual(hfi, 0.0)

    def test_fragmentation_no_fragmentation(self):
        """Test fragmentation computation with perfect preservation."""
        hfi = HyperedgeFragmentationIndex.compute(self.hn_intact,
                                                  self.hn_original)
        self.assertEqual(hfi, 0.0)

    def test_fragmentation_complete_destruction(self):
        """Test fragmentation computation with complete node loss."""
        hfi = HyperedgeFragmentationIndex.compute(self.hn_complete,
                                                  self.hn_original)
        self.assertEqual(hfi, 1.0)

    def test_fragmentation_partial_preservation(self):
        """Test fragmentation computation with partial node preservation."""
        hfi = HyperedgeFragmentationIndex.compute(self.hn_partial,
                                                  self.hn_original)

        # Calculate expected fragmentation
        # Hyperedge 0: [1,2,3] -> [1,2] preserved (2/3)
        # Hyperedge 1: [2,3,4] -> [2,4] preserved (2/3)
        # Hyperedge 2: [4,5,6] -> [4,5] preserved (2/3)
        # Hyperedge 3: [7,8] -> [7] preserved (1/2)
        expected_preservation = (2 / 3 + 2 / 3 + 2 / 3 + 1 / 2) / 4
        expected_fragmentation = 1.0 - expected_preservation

        self.assertAlmostEqual(hfi, expected_fragmentation, places=6)

    def test_fragmentation_single_node_remaining(self):
        """Test fragmentation computation with single node remaining."""
        hfi = HyperedgeFragmentationIndex.compute(self.hn_single,
                                                  self.hn_original)

        # Only node 3 remains, which appears in hyperedges 0 and 1
        # Hyperedge 0: [1,2,3] -> [3] preserved (1/3)
        # Hyperedge 1: [2,3,4] -> [3] preserved (1/3)
        # Hyperedge 2: [4,5,6] -> [] preserved (0/3)
        # Hyperedge 3: [7,8] -> [] preserved (0/2)
        expected_preservation = (1 / 3 + 1 / 3 + 0 / 3 + 0 / 2) / 4
        expected_fragmentation = 1.0 - expected_preservation

        self.assertAlmostEqual(hfi, expected_fragmentation, places=6)

    def test_fragmentation_bounds(self):
        """Test that values are properly bounded between 0 and 1."""
        # Test lower bound
        hfi_min = HyperedgeFragmentationIndex.compute(self.hn_intact,
                                                      self.hn_original)
        self.assertGreaterEqual(hfi_min, 0.0)
        self.assertLessEqual(hfi_min, 1.0)

        # Test upper bound
        hfi_max = HyperedgeFragmentationIndex.compute(self.hn_complete,
                                                      self.hn_original)
        self.assertGreaterEqual(hfi_max, 0.0)
        self.assertLessEqual(hfi_max, 1.0)

    def test_compute_detailed_analysis(self):
        """Test detailed fragmentation analysis."""
        detailed = HyperedgeFragmentationIndex.compute_detailed(
            self.hn_partial, self.hn_original)

        # Check required keys in detailed analysis
        expected_keys = {
            'fragmentation_index', 'total_hyperedges', 'intact_hyperedges',
            'partially_fragmented', 'completely_destroyed',
            'preservation_ratios',
            'avg_preservation_ratio'
        }
        self.assertEqual(set(detailed.keys()), expected_keys)

        # Check data types and bounds
        self.assertIsInstance(detailed['fragmentation_index'], float)
        self.assertIsInstance(detailed['total_hyperedges'], int)
        self.assertIsInstance(detailed['intact_hyperedges'], int)
        self.assertIsInstance(detailed['partially_fragmented'], int)
        self.assertIsInstance(detailed['completely_destroyed'], int)
        self.assertIsInstance(detailed['preservation_ratios'], list)

        self.assertGreaterEqual(detailed['fragmentation_index'], 0.0)
        self.assertLessEqual(detailed['fragmentation_index'], 1.0)
        self.assertEqual(detailed['total_hyperedges'], 4)

    def test_compute_detailed_empty_original(self):
        """Test detailed analysis with empty original hypergraph."""
        detailed = HyperedgeFragmentationIndex.compute_detailed(self.hn_empty,
                                                                self.hn_empty)

        self.assertEqual(detailed['fragmentation_index'], 0.0)
        self.assertEqual(detailed['total_hyperedges'], 0)
        self.assertEqual(detailed['intact_hyperedges'], 0)
        self.assertEqual(detailed['partially_fragmented'], 0)
        self.assertEqual(detailed['completely_destroyed'], 0)
        self.assertEqual(detailed['preservation_ratios'], [])

    def test_hyperedge_fragmentation_map(self):
        """Test individual hyperedge fragmentation mapping."""
        frag_map = HyperedgeFragmentationIndex.get_hyperedge_fragmentation_map(
            self.hn_partial, self.hn_original)

        # Check that all original hyperedges are mapped
        self.assertEqual(set(frag_map.keys()), {0, 1, 2, 3})

        # Check structure of each entry
        for he_id, info in frag_map.items():
            expected_keys = {
                'status', 'preservation_ratio', 'original_size',
                'preserved_size',
                'original_nodes', 'preserved_nodes'
            }
            self.assertEqual(set(info.keys()), expected_keys)

            # Check status values are valid
            self.assertIn(info['status'], {'intact', 'fragmented', 'destroyed'})

            # Check preservation ratio bounds
            self.assertGreaterEqual(info['preservation_ratio'], 0.0)
            self.assertLessEqual(info['preservation_ratio'], 1.0)

    def test_hyperedge_fragmentation_map_intact(self):
        """Test fragmentation mapping with intact hyperedges."""
        frag_map = HyperedgeFragmentationIndex.get_hyperedge_fragmentation_map(
            self.hn_intact, self.hn_original)

        # All hyperedges should be intact
        for he_id, info in frag_map.items():
            self.assertEqual(info['status'], 'intact')
            self.assertEqual(info['preservation_ratio'], 1.0)

    def test_hyperedge_fragmentation_map_destroyed(self):
        """Test fragmentation mapping with destroyed hyperedges."""
        frag_map = HyperedgeFragmentationIndex.get_hyperedge_fragmentation_map(
            self.hn_complete, self.hn_original)

        # All hyperedges should be destroyed
        for he_id, info in frag_map.items():
            self.assertEqual(info['status'], 'destroyed')
            self.assertEqual(info['preservation_ratio'], 0.0)

    def test_hyperedge_fragmentation_map_empty_hyperedges(self):
        """Test fragmentation mapping with empty hyperedges."""
        # Create hypergraph with empty hyperedge
        hn_with_empty = Hypernetwork()
        hn_with_empty.add_hyperedge(0, [])
        hn_with_empty.add_hyperedge(1, [1, 2])

        hn_perturbed = Hypernetwork()
        hn_perturbed.add_node(1)

        frag_map = HyperedgeFragmentationIndex.get_hyperedge_fragmentation_map(
            hn_perturbed, hn_with_empty)

        # Empty hyperedge should have 'empty' status
        self.assertEqual(frag_map[0]['status'], 'empty')
        self.assertEqual(frag_map[0]['preservation_ratio'], 0.0)

    def test_string_representations(self):
        """Test string and repr methods."""
        str_repr = str(self.fragmentation_metric)
        self.assertEqual(str_repr, "Hyperedge Fragmentation Index (HFI)")

        repr_str = repr(self.fragmentation_metric)
        expected_repr = (
            "HyperedgeFragmentationIndex(name='Hyperedge Fragmentation Index', "
            "symbol='HFI')")
        self.assertEqual(repr_str, expected_repr)


if __name__ == '__main__':
    unittest.main()
