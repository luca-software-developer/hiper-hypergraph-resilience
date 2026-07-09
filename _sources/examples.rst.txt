Examples
========

This page provides practical examples of using the HIPER library for
various hypernetwork analysis tasks.

Basic Hypernetwork Creation
----------------------------

Create a simple hypernetwork and perform basic operations:

.. code-block:: python

   from hiper import Hypernetwork

   # Create new hypernetwork instance
   hn = Hypernetwork()

   # Add hyperedges connecting multiple nodes
   hn.add_hyperedge(0, [1, 2, 3])
   hn.add_hyperedge(1, [2, 3, 4, 5])
   hn.add_hyperedge(2, [1, 4, 6])

   # Analyze network properties
   print(f"Network order: {hn.order()}")
   print(f"Network size: {hn.size()}")
   print(f"Average degree: {hn.avg_deg():.2f}")

   # Query node connectivity
   neighbors = hn.get_neighbors(2)
   hyperedges = hn.get_hyperedges(2)

Loading Datasets
----------------

Load standard datasets for analysis:

.. code-block:: python

   from hiper import DataFile, Dataset, load_config

   # Load configuration and dataset
   config = load_config()
   datafile = DataFile('data/Algebra.txt')
   dataset = Dataset('Algebra', datafile)

   # Access constructed hypernetwork
   hypernetwork = dataset.get_hypernetwork()
   hypernetwork.print_info()

Attack Simulation
-----------------

Execute individual attacks with restoration:

.. code-block:: python

   from hiper import (
       HypernetworkSimulator, AddNodeAttack, RemoveNodeAttack,
       AttackSequence, Hypernetwork
   )

   # Create simulator and set target network
   hn = Hypernetwork()
   hn.add_hyperedge(0, [1, 2, 3])
   hn.add_hyperedge(1, [2, 3, 4, 5])

   simulator = HypernetworkSimulator('security_analysis')
   simulator.set_hypernetwork(hn)

   # Execute individual attack with restoration
   attack = RemoveNodeAttack('remove_critical_node', 2)
   result = simulator.simulate_attack(attack)

   print(f"Attack success: {result['success']}")
   print(f"Network impact: {result['changes']['order_change']} nodes")

Attack Sequences
----------------

Create and execute coordinated attack sequences:

.. code-block:: python

   from hiper import (
       HypernetworkSimulator, AddNodeAttack, RemoveNodeAttack,
       AttackSequence, Hypernetwork
   )

   # Create hypernetwork
   hn = Hypernetwork()
   hn.add_hyperedge(0, [1, 2, 3])
   hn.add_hyperedge(1, [2, 3, 4, 5])

   # Create simulator
   simulator = HypernetworkSimulator('coordinated_attack')
   simulator.set_hypernetwork(hn)

   # Create coordinated attack sequence
   sequence = AttackSequence('coordinated_attack')
   sequence.add_attack(AddNodeAttack('add_decoy', 10))
   sequence.add_attack(RemoveNodeAttack('remove_target', 1))

   # Execute sequence with detailed analysis
   sequence_result = simulator.simulate_sequence(sequence)
   execution_stats = sequence_result['execution_stats']
   print(f"Success rate: {execution_stats['success_rate']:.2%}")

Resilience Analysis
-------------------

Perform resilience experiments using TOPSIS:

.. code-block:: python

   from hiper import Hypernetwork
   from hiper.metrics.experiments import ResilienceExperiment

   # Create or load hypernetwork
   hn = Hypernetwork()
   # ... add hyperedges ...

   # Create experiment
   experiment = ResilienceExperiment(s=1)

   # Run node removal experiment
   results = experiment.run_node_removal_experiment(
       hypernetwork=hn,
       removal_percentages=[1, 2, 5, 10, 25],
       random_trials=10
   )

   # Analyze results
   print("Original metrics:", results['original_metrics'])
   print("Random removal:", results['random_removal'])
   print("TOPSIS top removal:", results['topsis_top_removal'])

TOPSIS Node Ranking
-------------------

Rank nodes by importance using TOPSIS:

.. code-block:: python

   from hiper import Hypernetwork
   from hiper.metrics.topsis import TopsisNodeRanker

   # Create hypernetwork
   hn = Hypernetwork()
   # ... add hyperedges ...

   # Create ranker
   ranker = TopsisNodeRanker()

   # Get top 10% most critical nodes
   top_nodes = ranker.get_top_nodes(hn, percentage=10)
   print(f"Most critical nodes: {top_nodes}")

   # Get bottom 10% least critical nodes
   bottom_nodes = ranker.get_bottom_nodes(hn, percentage=10)
   print(f"Least critical nodes: {bottom_nodes}")

MCDM Methods Comparison
------------------------

Compare different MCDM methods for node ranking:

.. code-block:: python

   from hiper import Hypernetwork
   from hiper.metrics.topsis import TopsisNodeRanker
   from hiper.metrics.wsm import WSMNodeRanker
   from hiper.metrics.moora import MOORANodeRanker

   # Create hypernetwork
   hn = Hypernetwork()
   # ... add hyperedges ...

   # Create different rankers
   topsis = TopsisNodeRanker()
   wsm = WSMNodeRanker()
   moora = MOORANodeRanker()

   # Compare rankings
   topsis_ranking = topsis.rank_nodes(hn)
   wsm_ranking = wsm.rank_nodes(hn)
   moora_ranking = moora.rank_nodes(hn)

   print(f"TOPSIS top node: {topsis_ranking[0]}")
   print(f"WSM top node: {wsm_ranking[0]}")
   print(f"MOORA top node: {moora_ranking[0]}")

Custom Metrics
--------------

Compute custom structural metrics:

.. code-block:: python

   from hiper import Hypernetwork
   from hiper.metrics.connectivity import HypergraphConnectivity
   from hiper.metrics.distance import HypergraphDistance

   # Create hypernetwork
   hn = Hypernetwork()
   # ... add hyperedges ...

   # Compute connectivity
   conn = HypergraphConnectivity()
   hg_conn = conn.compute_hypergraph_connectivity(hn)
   he_conn = conn.compute_hyperedge_connectivity(hn)

   print(f"Hypergraph connectivity: {hg_conn}")
   print(f"Hyperedge connectivity: {he_conn}")

   # Compute distances
   dist = HypergraphDistance()
   is_connected = dist.is_connected(hn)
   diameter = dist.diameter(hn)

   print(f"Is connected: {is_connected}")
   print(f"Diameter: {diameter}")
