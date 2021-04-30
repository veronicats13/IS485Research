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
            "Susceptible": 0, #change to inactive
            "Infected": 1, #change to active
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

        # can remove this step
        # only tells us that the activated nodes are active while the other nodes are inactive
        '''
        {'iteration': 0, 'status': {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0, 128: 0, 129: 0, 130: 0, 131: 0, 132: 0, 133: 0, 134: 0, 135: 0, 136: 0, 137: 0, 138: 0, 139: 0, 140: 0, 141: 0, 142: 0, 143: 0, 144: 0, 145: 0, 146: 0, 147: 0, 148: 0, 149: 0}, 
        'node_count': {0: 149, 1: 1}, 'status_delta': {0: 0, 1: 0}}
        '''
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
        # can consider "for u in activated_nodes:"
        # saves time especially in a big graph
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
                        if 'threshold' in self.params['edges']: # if edge has a threshold
                            if key in self.params['edges']['threshold']: # if key (u, v) in params... but why would it be in here? oh cos edges if from node to node so tuple (u , v)
                                threshold = self.params['edges']['threshold'][key] # replace key?
                            elif (v, u) in self.params['edges']['threshold'] and not self.graph.directed: # direction affects this. v to u instead of u to v. yup this
                                threshold = self.params['edges']['threshold'][(v, u)] # similarly put in key in (but opposite direction)
                                # oh this is the actual threshold used below for the flip i think
                        
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