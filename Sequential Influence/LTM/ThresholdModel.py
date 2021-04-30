from DiffusionModel import DiffusionModel
import future.utils

__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class ThresholdModel(DiffusionModel):
    """
        Node Parameters to be specified via ModelConfig

       :param threshold: The node threshold. If not specified otherwise a value of 0.1 is assumed for all nodes.
    """

    def __init__(self, graph, seed=None):
        """
             Model Constructor

             :param graph: A networkx graph object
         """
        super(self.__class__, self).__init__(graph, seed)
        self.available_statuses = {
            "Susceptible": 0,
            "Infected": 1
        }

        self.parameters = {
            "model": {},
            "nodes": {
                "threshold": {
                    "descr": "Node threshold",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.1
                }
            },
            "edges": {
                "weight": {
                    "descr": "Edge weight", 
                    "range" : [0, 1], 
                    "optional": True, 
                    "default": 0.1
                }
            },
        }

        self.name = "Threshold"

    def iteration(self, node_status=True):
        """
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        
        self.clean_initial_status(list(self.available_statuses.values()))
        actual_status = {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}

        for u in self.graph.nodes:
            neighbors = []

            if actual_status[u] == 1:
                continue
            
            neighbors = list(self.graph.neighbors(u))

            if self.graph.directed:
                neighbors = list(self.graph.predecessors(u))

            # activate if the total weight of its ACTIVE neighbours is at least threshold
            if len(neighbors) > 0:
                total_weight = 0

                for neighbor in neighbors:

                    # find active neighbours
                    if self.status[neighbor] == 1: 
                        # print(u, neighbor)
                        if (u, neighbor) in list(self.graph.edges): 
                            total_weight += self.params['edges']['weight'][(u, neighbor)] 
                
                # compare total weight with threshold
                if total_weight >= self.params['nodes']['threshold'][u]: 
                    actual_status[u] = 1
                
        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1
        for key in status_delta: 
            if status_delta[key] != 0: 
                self.stop = False
            else: 
                self.stop = True

        active_set_size = node_count.copy()[1]
        status = delta.copy()
        
        return active_set_size, status
