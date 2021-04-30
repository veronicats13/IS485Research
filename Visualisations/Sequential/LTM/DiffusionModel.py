import abc
import warnings
import numpy as np
import past.builtins
import future.utils
import six
from netdispatch import AGraph
import tqdm
import random
import math

__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class ConfigurationException(Exception):
    """Configuration Exception"""


@six.add_metaclass(abc.ABCMeta)
class DiffusionModel(object):
    """
        Partial Abstract Class that defines Diffusion Models
    """
    # __metaclass__ = abc.ABCMeta

    def __init__(self, graph, seed=None):
        """
            Model Constructor

            :param graph: A networkx graph object
        """

        np.random.seed(seed)

        self.discrete_state = True

        self.params = {
            'nodes': {},
            'edges': {},
            'model': {},
            'status': {}
        }

        self.available_statuses = {
            "Susceptible": 0,
            "Infected": 1,
            "Recovered": 2, #remove
            "Blocked": -1 #remove
        }

        self.name = ""

        self.parameters = {
            "model": {},
            "nodes": {},
            "edges": {}
        }

        self.actual_iteration = 0
        self.graph = AGraph(graph)
        self.status = {n: 0 for n in self.graph.nodes}
        self.attempt = {n: 0 for n in self.graph.nodes}
        self.initial_status = {}
        self.stop = False

    def __validate_configuration(self, configuration):
        """
        Validate the consistency of a Configuration object for the specific model

        :param configuration: a Configuration object instance
        """
        if "Infected" not in self.available_statuses:
            raise ConfigurationException("'Infected' status not defined.")

        # Checking mandatory parameters
        omp = set([k for k in list(self.parameters['model'].keys()) if not self.parameters['model'][k]['optional']])
        onp = set([k for k in list(self.parameters['nodes'].keys()) if not self.parameters['nodes'][k]['optional']])
        oep = set([k for k in list(self.parameters['edges'].keys()) if not self.parameters['edges'][k]['optional']])
        # print(onp)
        # print(oep)

        mdp = set(configuration.get_model_parameters().keys())
        ndp = set(configuration.get_nodes_configuration().keys())
        edp = set(configuration.get_edges_configuration().keys())
        # print(ndp) #somehow in second influencer selection onwards the param attempt comes up even tho i reset the model and config
        # print(edp)

        if len(omp) > 0:
            if len(omp & mdp) != len(omp):
                raise ConfigurationException({"message": "Missing mandatory model parameter(s)", "parameters": omp-mdp})

        if len(onp) > 0:
            if len(onp & ndp) != len(onp):
                raise ConfigurationException({"message": "Missing mandatory node parameter(s)", "parameters": onp-ndp})

        if len(oep) > 0:
            if len(oep & edp) != len(oep):
                raise ConfigurationException({"message": "Missing mandatory edge parameter(s)", "parameters": oep-edp})

        # Checking optional parameters
        omp = set([k for k in list(self.parameters['model'].keys()) if self.parameters['model'][k]['optional']])
        onp = set([k for k in list(self.parameters['nodes'].keys()) if self.parameters['nodes'][k]['optional']])
        oep = set([k for k in list(self.parameters['edges'].keys()) if self.parameters['edges'][k]['optional']])
        # print(onp)
        # print(oep)

        if len(omp) > 0:
            for param in omp:
                if param not in mdp:
                    configuration.add_model_parameter(param, self.parameters['model'][param]['default'])

        if len(onp) > 0:
            for param in onp:
                # print(param)
                # print(ndp)
                if param not in ndp:
                    
                    for nid in self.graph.nodes:
                        configuration.add_node_configuration(param, nid, self.parameters['nodes'][param]['default'])
                        # print(nid)
                        # print(self.parameters['nodes'][param]['default'])
                        # print('node done')

        if len(oep) > 0:
            for param in oep:
                if param not in edp:
                    for eid in self.graph.edges:
                        configuration.add_edge_configuration(param, eid, self.parameters['edges'][param]['default'])
                        # print(eid)
                        # print(self.parameters['edges'][param]['default'])
                        # print('edge done')
                        

        # Checking initial simulation status
        sts = set(configuration.get_model_configuration().keys())
        if self.discrete_state and "Infected" not in sts and "fraction_infected" not in mdp \
                and "percentage_infected" not in mdp:
            warnings.warn('Initial infection missing: a random sample of 5% of graph nodes will be set as infected')
            self.params['model']["fraction_infected"] = 0.05

    def set_initial_status(self, configuration):
        """
        Set the initial model configuration

        :param configuration: a ```ndlib.models.ModelConfig.Configuration``` object
        """

        self.__validate_configuration(configuration)

        nodes_cfg = configuration.get_nodes_configuration()
        # Set additional node information

        for param, node_to_value in future.utils.iteritems(nodes_cfg):
            if len(node_to_value) < len(self.graph.nodes):
                raise ConfigurationException({"message": "Not all nodes have a configuration specified"})

            self.params['nodes'][param] = node_to_value

        edges_cfg = configuration.get_edges_configuration()
        # Set additional edges information
        for param, edge_to_values in future.utils.iteritems(edges_cfg):
            if len(edge_to_values) == len(self.graph.edges):
                self.params['edges'][param] = {}
                for e in edge_to_values:
                    self.params['edges'][param][e] = edge_to_values[e]

        # Set initial status
        model_status = configuration.get_model_configuration()

        for param, nodes in future.utils.iteritems(model_status):
            self.params['status'][param] = nodes
            for node in nodes:
                self.status[node] = self.available_statuses[param]

        # Set model additional information
        model_params = configuration.get_model_parameters()
        for param, val in future.utils.iteritems(model_params):
            self.params['model'][param] = val

        # Handle initial infection
        if 'Infected' not in self.params['status']:
            if 'percentage_infected' in self.params['model']:
                self.params['model']['fraction_infected'] = self.params['model']['percentage_infected']
            if 'fraction_infected' in self.params['model']:
                number_of_initial_infected = self.graph.number_of_nodes() * float(self.params['model']['fraction_infected'])
                if number_of_initial_infected < 1:
                    warnings.warn(
                        "The fraction_infected value is too low given the number of nodes of the selected graph: a "
                        "single node will be set as infected")
                    number_of_initial_infected = 1

                available_nodes = [n for n in self.status if self.status[n] == 0]
                sampled_nodes = np.random.choice(available_nodes, int(number_of_initial_infected), replace=False)
                for k in sampled_nodes:
                    self.status[k] = self.available_statuses['Infected']

        self.initial_status = self.status

    def clean_initial_status(self, valid_status=None):
        """
        Check the consistency of initial status
        :param valid_status: valid node configurations
        """
        for n, s in future.utils.iteritems(self.status):
            if s not in valid_status:
                self.status[n] = 0

    # def iteration_bunch(self, bunch_size, node_status=True):
    def iteration_bunch(self, node_status=True):
        """
        Execute a bunch of model iterations

        :param bunch_size: the number of iterations to execute
        :param node_status: if the incremental node status has to be returned.

        :return: a list containing for each iteration a dictionary {"iteration": iteration_id, "status": dictionary_node_to_status}
        """
        system_status = []
        all_activated_nodes = []
        step_activated_nodes = []
        while not self.stop:
            its = self.iteration(node_status)
            all_activated_nodes.extend(list(its[1].keys()))
            step_activated_nodes.append(list(its[1].keys()))
            active_set_size = its[0]
        return active_set_size, all_activated_nodes, step_activated_nodes
            
    def random_deactivation(self, activated_nodes):
        """
        Randomly deactivate nodes from the set of activated nodes
        """
        if activated_nodes != []:
            deactivated_nodes = []
            k = random.randint(0, math.ceil(len(activated_nodes) * 0.05))
            deactivated_nodes = random.sample(activated_nodes, k = k)
            print('Number of deactivations: ' + str(k))
            print('Deactivated Nodes: ' + str(deactivated_nodes))
            for node in deactivated_nodes: 
                self.status[node] = 0
                activated_nodes.remove(node)
        return len(activated_nodes), activated_nodes,  deactivated_nodes

    def mg_reset(self, temp_activated_nodes):
        # mg reset for mg testing
        #reset nodes that were activated during marginal gain testing
        # need to reset attempted to 0 as well
        # reset self.stop to false so can keep on testing
        for n in temp_activated_nodes:
            self.status[n] = 0
            # self.params['nodes']['attempt'][n] = 0
        self.stop = False #so can run again
        # return     
        # return self.status

    def is_reset(self):
        # is reset for influence spread
        # reset self.stop so can proceed with subsequent mg testing and is 
        self.stop = False

    def get_info(self):
        """
        Describes the current model parameters (nodes, edges, status)

        :return: a dictionary containing for each parameter class the values specified during model configuration
        """
        info = {k: v for k, v in future.utils.iteritems(self.params) if k not in ['nodes', 'edges', 'status']}
        if 'infected_nodes' in self.params['status']:
            info['selected_initial_infected'] = True
        return info['model']

    def reset(self, infected_nodes=None):
        """
        Reset the simulation setting the actual status to the initial configuration.
        """
        self.actual_iteration = 0

        if infected_nodes is not None:
            for n in self.status:
                self.status[n] = 0
            for n in infected_nodes:
                self.status[n] = self.available_statuses['Infected']
            self.initial_status = self.status
            self.stop = Falses

        else:
            if 'percentage_infected' in self.params['model']:
                self.params['model']['fraction_infected'] = self.params['model']['percentage_infected']
            if 'fraction_infected' in self.params['model']:
                for n in self.status:
                    self.status[n] = 0
                number_of_initial_infected = self.graph.number_of_nodes() * float(self.params['model']['fraction_infected'])
                available_nodes = [n for n in self.status if self.status[n] == 0]
                sampled_nodes = np.random.choice(available_nodes, int(number_of_initial_infected), replace=False)

                for k in sampled_nodes:
                    self.status[k] = self.available_statuses['Infected']

                self.initial_status = self.status
            else:
                self.status = self.initial_status

        return self

    def get_model_parameters(self):
        return self.parameters

    def get_name(self):
        return self.name

    def get_status_map(self):
        """
        Specify the statuses allowed by the model and their numeric code

        :return: a dictionary (status->code)
        """
        return self.available_statuses

    @abc.abstractmethod
    def iteration(self, node_status=True):
        """
        Execute a single model iteration

        :param node_status: if the incremental node status has to be returned.

        :return: Iteration_id,
                 (optional) Incremental node status (dictionary node->status),
                 Status count (dictionary status->node count),
                 Status delta (dictionary status->node delta)
        """
        pass

    @staticmethod
    def check_status_similarity(actual, previous):
        """
        Evaluate similarity among statuses

        :param actual: actual status
        :param previous: previous status
        :return: True if the two statuses are the same, False otherwise
        """
        for n, v in future.utils.iteritems(actual):
            if n not in previous:
                return False
            if previous[n] != actual[n]:
                return False
        return True

    def status_delta(self, actual_status):
        """
        Compute the point-to-point variations for each status w.r.t. the previous system configuration

        :param actual_status: the actual simulation status
        :return: node that have changed their statuses (dictionary status->nodes),
                 count of actual nodes per status (dictionary status->node count),
                 delta of nodes per status w.r.t the previous configuration (dictionary status->delta)
        """
        actual_status_count = {}
        old_status_count = {}
        delta = {}
        for n, v in future.utils.iteritems(self.status):
            if v != actual_status[n]:
                delta[n] = actual_status[n]

        for st in list(self.available_statuses.values()):
            actual_status_count[st] = len([x for x in actual_status if actual_status[x] == st])
            old_status_count[st] = len([x for x in self.status if self.status[x] == st])

        status_delta = {st: actual_status_count[st] - old_status_count[st] for st in actual_status_count}

        return delta, actual_status_count, status_delta

    def status_delta_continuous(self, actual_status):
        """
        Compute the point-to-point variations for each status w.r.t. the previous system configuration

        Should be used for continuous statuses instead of discrete values

        :param actual_status: the actual simulation status
        :return: nodes that have changed their statuses (dictionary status->nodes),
                    count of actual nodes per status (dictionary status->node count),
                    delta of nodes per status w.r.t the previous configuration (dictionary status->delta)
        """
        delta = {}
        status_delta = {}
        for n, v in future.utils.iteritems(self.status):
            # print(n)
            # print(v)
            delta[n] = {}
            status_delta[n] = {}
            for var, val in list(v.items()):
                if val != actual_status[n][var]:
                    delta[n][var] = actual_status[n][var]
                    status_delta[n][var] = actual_status[n][var] - val
            if len(list(delta[n].values())) == 0:
                del delta[n]
            if len(list(status_delta[n].values())) == 0:
                del status_delta[n]

        return delta, status_delta


    def build_trends(self, iterations):
        """
        Build node status and node delta trends from model iteration bunch

        :param iterations: a set of iterations
        :return: a trend description
        """
        status_delta = {status: [] for status in list(self.available_statuses.values())}
        node_count = {status: [] for status in list(self.available_statuses.values())}

        for it in iterations:
            for st in list(self.available_statuses.values()):
                try:
                    status_delta[st].append(it['status_delta'][st])
                    node_count[st].append(it['node_count'][st])
                except:
                    status_delta[st].append(it['status_delta'][str(st)])
                    node_count[st].append(it['node_count'][str(st)])

        return [{"trends": {"node_count": node_count, "status_delta": status_delta}}]