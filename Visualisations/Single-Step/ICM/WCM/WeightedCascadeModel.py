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

        # if self.actual_iteration == 0:
        #     self.actual_iteration += 1
        #     delta, node_count, status_delta = self.status_delta(actual_status)
        #     if node_status:
        #         return {"iteration": 0, "status": actual_status.copy(),
        #                 "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        #     else:
        #         return {"iteration": 0, "status": {},
        #                 "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        count_attempts = 0
        # print(self.params['nodes']['attempt'])
        for u in self.graph.nodes: # for each node
            if self.status[u] != 1: # only select nodes with status = 1 (infected/active)
                continue
            if self.params['nodes']['attempt'][u] != 0: # and attempt = 0 (no previous attempts)
                continue
            # print("go")

            neighbors = list(self.graph.neighbors(u))  # neighbors and successors (in DiGraph) produce the same result
            # get neighbors of this infected/active node

            # Standard threshold
            if len(neighbors) > 0:  
                for v in neighbors: # for each neighbor
                    if actual_status[v] == 0: # if their status = 0 (susceptible/inactive)
                        key = (u, v) # key = (active node, inactive node) or (infected node, susceptible node)

                        # Individual specified thresholds
                        # if 'threshold' in self.params['edges']: # if edge has a threshold
                        #     if key in self.params['edges']['threshold']: # if key (u, v) in params... but why would it be in here? oh cos edges if from node to node so tuple (u , v)
                        #         threshold = self.params['edges']['threshold'][key] # replace key?
                        #     elif (v, u) in self.params['edges']['threshold'] and not self.graph.directed: # direction affects this. v to u instead of u to v. yup this
                        #         threshold = self.params['edges']['threshold'][(v, u)] # similarly put in key in (but opposite direction)
                                # oh this is the actual threshold used below for the flip i think
                        
                        degree_v = len(list(self.graph.neighbors(v)))
                        threshold = 1.0/degree_v
                        flip = np.random.random_sample() # random float in half-open interval [0.0, 1.0)
                        if flip <= threshold: # if less than threshold 
                            actual_status[v] = 1 # neighbor becomes infected/active
                        # actual_status[v] = 1 # probability activated is 1
            self.params['nodes']['attempt'][u] = 1
            count_attempts += 1

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        # print("self status")
        # print(self.status)
        self.actual_iteration += 1
        if count_attempts == 0:
            self.stop = True

        # this one remains but change to ensure the output fits to what we want
        # we only want the active set size at the end of the iteration
        # can get from the last 'node_count': {0: inactive_set_size, 1: active_set_size}
        # if node_status:
        #     return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
        #             "node_count": node_count.copy(), "status_delta": status_delta.copy(), 'active_set_size': node_count.copy()[1]}
        # else:
        #     return {"iteration": self.actual_iteration - 1, "status": {},
        #             "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        
        active_set_size = node_count.copy()[1]
        status = delta.copy()
        # print(status)
        
        return active_set_size, status

                        
