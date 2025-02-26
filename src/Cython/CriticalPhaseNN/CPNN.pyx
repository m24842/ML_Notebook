import random
cimport cython
import numpy as np
from itertools import product

cdef class CriticalPhaseNeuralNetwork:
    cdef public int width
    cdef public int dimensions
    cdef public dict nodes
    cdef public list input_nodes_coords
    cdef public int nodes_active
    cdef public double connection_prob

    def __cinit__(self, int width = 10, int dimensions = 2, int num_input_nodes = 0, double connection_prob = 0.5, str model_path = ""):
        if model_path.endswith(".npz"): self.load_model(model_path)
        else:
            self.connection_prob = connection_prob
            self.width = width
            self.dimensions = dimensions
            num_nodes = width ** dimensions
            assert num_input_nodes <= num_nodes, "Invalid network parameters"
            self.nodes = {}
            self.input_nodes_coords = []
            self.nodes_active = 0
            self.create_nodes(width, dimensions, num_input_nodes)

    def __reduce__(self):
        return (CriticalPhaseNeuralNetwork, (self.width, self.dimensions, self.nodes, self.input_nodes_coords, self.connection_prob))

    cpdef void load_model(self, str model_path):
        loaded_model = np.load(model_path, allow_pickle=True)
        self.width = loaded_model["width"]
        self.dimensions = loaded_model["dimensions"]
        self.nodes = loaded_model["nodes"].item()
        self.input_nodes_coords = [tuple(coords) for coords in loaded_model["input_nodes_coords"].tolist()]
        self.connection_prob = loaded_model["connection_prob"]

    cpdef void save_model(self, str model_path):
        # Reset node states before saving
        self.reset()
        np.savez(model_path,
            width = self.width,
            dimensions = self.dimensions,
            nodes = self.nodes,
            input_nodes_coords = self.input_nodes_coords,
            connection_prob = self.connection_prob,
            allow_pickle=True
        )

    def create_nodes(self, int width, int dimensions, int num_input_nodes):
        cdef int num_nodes = width ** dimensions

        # Create the nodes in the network
        cdef list all_coords = list(product(range(width), repeat=dimensions))
        cdef tuple coords
        for coords in all_coords:
            self.nodes[coords] = Node(coords)

        # Offsets
        cdef list offsets = list(product([-1, 0, 1], repeat=dimensions))
        # Link all nodes to their neighbors
        cdef tuple current_coords
        cdef Node current_node
        cdef list neighboring_coords
        # Misc.
        cdef int i, index
        cdef double[:] random_weights
        # Link all nodes to their neighbors
        for current_coords, current_node in self.nodes.items():
            neighboring_coords=[tuple(x + y for x, y in zip(current_coords, offset)) for offset in offsets if tuple(x + y for x, y in zip(current_coords, offset)) != current_coords]
            random_weights = np.random.normal(loc=0, scale=1, size=len(neighboring_coords))
            random_connections = np.random.choice([0, 1], size=len(neighboring_coords), p=[1 - self.connection_prob, self.connection_prob])
            for i, coords in enumerate(neighboring_coords):
                if coords in self.nodes and random_connections[i] == 1:
                    current_node.link(self.nodes[coords], random_weights[i], True)

        # Randomly select input nodes
        # cdef random_coords = random.sample(range(num_nodes), num_input_nodes)
        # for i, index in enumerate(random_coords):
        #     self.input_nodes_coords.append(all_coords[index])
        #     self.nodes[all_coords[index]].is_input_node = True

        # Evenly select input nodes
        input_nodes_per_dim = round(pow(num_input_nodes, 1 / dimensions) + 0.5)
        spacing = width // input_nodes_per_dim
        padding = (width - spacing * input_nodes_per_dim) // 2

        all_coords = [range(padding + spacing//2, width, spacing) for _ in range(dimensions)]
        all_coords = list(product(*all_coords))

        for i, coords in enumerate(all_coords):
            if i >= num_input_nodes: break
            self.input_nodes_coords.append(coords)
            self.nodes[coords].is_input_node = True

    cpdef tuple feedforward(self):
        self.nodes_active = 0
        cdef int[:] output = cython.view.array(shape=(len(self.input_nodes_coords),), itemsize=cython.sizeof(int), format='i') if len(self.input_nodes_coords) > 0 else np.array([], dtype=np.int32)
        cdef dict net_state = {}
        cdef Node current_node
        cdef tuple coords
        for coords, current_node in self.nodes.items():
            current_node.update_state()
            net_state[(coords)] = (current_node.state, current_node.is_input_node)
            if current_node.state == 1:
                self.nodes_active += 1
        for coords, current_node in self.nodes.items():
            current_node.feedforward(self)
        cdef int i
        cdef Node node
        for i, coords in enumerate(self.input_nodes_coords):
            output[i] = self.nodes[(coords)].state
        return (net_state, output)

    cpdef void load_input(self, int[:] input_data):
        assert len(input_data) == len(self.input_nodes_coords), f"Invalid input data size: {len(input_data)} != {len(self.input_nodes_coords)}"
        # Look for all excitatory and inhibitory neurons and randomly connect them to any firing input neurons
        if input_data.count(1) == 0: return
        cdef int i, j
        cdef Node node
        cdef tuple coords
        cdef list non_dormant_nodes = []
        for coords, node in self.nodes.items():
            if node.state != 0: non_dormant_nodes.append(node)
        cdef double[:] random_weights = np.random.normal(loc=0, scale=1, size=len(non_dormant_nodes) * len(input_data))
        cdef double sign
        for i, coords in enumerate(self.input_nodes_coords):
            if input_data[i] == 1:
                for j, node in enumerate(non_dormant_nodes):
                    sign = 1.0 if node.state == 1 else -1.0
                    node.link(self.nodes[(coords)], sign * abs(random_weights[i * len(non_dormant_nodes) + j]))
        # Load input data into input nodes
        for i, coords in enumerate(self.input_nodes_coords):
            if input_data[i] == 1: self.nodes[(coords)].nextState = input_data[i]
        
    cpdef list inference(self, int[:] input_data, double max_time_steps = 0, bint crop = False):
        if max_time_steps == 0:
            max_time_steps = np.inf
        cdef list states_and_outputs = []
        cdef int t
        cdef tuple state_and_output
        cdef bint same_state
        self.load_input(input_data)
        for t in range(int(max_time_steps)):
            state_and_output = self.feedforward()
            states_and_outputs.append(state_and_output)
            if crop and self.nodes_active == 0:
                break
        return states_and_outputs

    # Reset the network
    cpdef void reset(self):
        self.nodes_active = 0
        cdef Node node
        cdef tuple coords
        for coords, node in self.nodes.items():
            node.reset()

    cpdef int count_edges(self):
        cdef int total_edges = 0
        cdef Node node
        cdef tuple coords
        for coords, node in self.nodes.items():
                total_edges += len(node.outputs)
        return total_edges

cdef class Node:
    cdef public tuple coords
    cdef public bint is_input_node
    cdef public int refractory_tick # How long the node will stay in refractory
    cdef public int refractory_period # Max refractory period
    cdef public int state # -1, 0, or 1 for inhibitory, inactive, or excitatory
    cdef public double nextState
    cdef public list inputs
    cdef public list outputs

    def __cinit__(self, tuple coords, bint is_input_node = False, int refractory_tick = 0, int refractory_period = 50, int state = 0, double nextState = 0, list inputs = None, list outputs = None):
        self.coords = coords
        self.is_input_node = is_input_node
        self.refractory_tick = refractory_tick
        self.refractory_period = refractory_period
        self.state = state
        self.nextState = nextState
        if inputs is None:
            self.inputs = []
        else:
            self.inputs = inputs
        if outputs is None:
            self.outputs = []
        else:
            self.outputs = outputs

    def __reduce__(self):
        return (Node, (self.coords, self.is_input_node, self.refractory_tick, self.refractory_period, self.state, self.nextState, self.inputs, self.outputs))

    cpdef void link(self, Node target, double weight = 0.0, bint check_for_dupes = True):
        cdef bint connection_exists = False
        cdef Edge edge
        if check_for_dupes:
            for edge in self.outputs:
                if edge.target_coords == target.coords:
                    connection_exists = True
                    break
        if not connection_exists:
            edge = Edge(self.coords, target.coords, weight)
            self.outputs.append(edge)
            target.inputs.append(edge)

    cpdef void update_state(self):
        if self.refractory_tick == 0:
            if self.nextState > 0:
                self.state = 1
            elif self.nextState == 0:
                self.state = 0
            else:
                self.state = -1
        else:
            self.state = 0
        self.nextState = 0
    
    cpdef void feedforward(self, CriticalPhaseNeuralNetwork network):
        cdef Edge edge
        if self.refractory_tick == 0 and self.state != 0:
            for edge in self.outputs:
                edge.feedforward(network)
        if self.state == 1 or self.state == -1:
            self.refractory_tick = self.refractory_period
        elif self.refractory_tick > 0:
            self.refractory_tick -= 1

    cpdef void reset(self):
        self.refractory_tick = 0
        self.state = 0
        self.nextState = 0

cdef class Edge:
    cdef public double weight
    cdef public tuple source_coords
    cdef public tuple target_coords

    def __cinit__(self, tuple source_coords, tuple target_coords, double weight):
        self.weight = weight
        self.source_coords = source_coords
        self.target_coords = target_coords

    def __reduce__(self):
        return (Edge, (self.source_coords, self.target_coords, self.weight))

    # Propagate source node signal depending on weight
    cpdef void feedforward(self, CriticalPhaseNeuralNetwork network):
        network.nodes[self.target_coords].nextState += self.weight * network.nodes[self.source_coords].state