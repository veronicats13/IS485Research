from DiffusionModel import DiffusionModel
import numpy as np
import future.utils

__author__ = 'Giulio Rossetti'
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class IndependentCascadesModel(DiffusionModel):
    """
        Edge Parameters to be specified via ModelConfig

        :param threshold: The edge threshold. As default a value of 0.1 is assumed for all edges.
    """

    def __init__(self, graph, seed=None):
        """
             Model Constructor

             :param graph: A networkx graph object
         """
        super(self.__class__, self).__init__(graph, seed)
        self.available_statuses = {
            "Susceptible": 0,
            "Infected": 1,
        }

        self.parameters = {
            "model": {},
            "nodes": {
                "attempt": {
                    "descr": "Node attempts",
                    "optional": True,
                    "default": 0
                }
            },
            "edges": {
                "threshold": {
                    "descr": "Edge threshold",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.1
                }
            },
        }

        self.name = "Independent Cascades"

    def iteration(self, node_status=True):
        """
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        self.clean_initial_status(list(self.available_statuses.values()))
        actual_status = {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}

        count_attempts = 0
        for u in self.graph.nodes: 
            if self.status[u] != 1: 
                continue
            if self.params['nodes']['attempt'][u] != 0:
                continue

            neighbors = list(self.graph.neighbors(u))
            if len(neighbors) > 0:  
                for v in neighbors:
                    if actual_status[v] == 0:
                        key = (u, v)

                        if 'threshold' in self.params['edges']: 
                            if key in self.params['edges']['threshold']:
                                threshold = self.params['edges']['threshold'][key]
                            elif (v, u) in self.params['edges']['threshold'] and not self.graph.directed: 
                                threshold = self.params['edges']['threshold'][(v, u)] 
                        
                        flip = np.random.random_sample() 
                        if flip <= threshold:
                            actual_status[v] = 1

            self.params['nodes']['attempt'][u] = 1
            count_attempts += 1

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1
        if count_attempts == 0:
            self.stop = True
        
        active_set_size = node_count.copy()[1]
        status = delta.copy()
        
        return active_set_size, status