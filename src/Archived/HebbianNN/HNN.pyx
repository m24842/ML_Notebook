import random
from libc.math cimport exp
import numpy as np

cdef double CHANGE_RATE = 0.05

cdef class HebbianNeuralNetwork:
    cdef public int[:] shape
    cdef public list layers
    cdef public double[:] normalized_input
    cdef public double[:] normalized_expected_output
    
    def __cinit__(self, int[:] nodes_per_layer, list layer_activations):
        assert len(nodes_per_layer) == len(layer_activations), "Number of layers and activation functions do not match"
        self.layers = [None] * len(nodes_per_layer)
        self.shape = nodes_per_layer
        self.create_layers(nodes_per_layer, layer_activations)
    
    cpdef void create_layers(self, int[:] nodes_per_layer, list layer_activations):
        for i, (num_nodes, activation_type) in enumerate(zip(nodes_per_layer, layer_activations)):
            self.layers[i] = Layer(num_nodes, activation_type)
        for i in range(len(self.layers) - 1):
            # Link forward
            self.layers[i].link(self.layers[i + 1])
    
    # Update the states of the nodes in the network by one timestep
    cpdef list forward_propagation(self):
        cdef list output = []
        # Feed the input data into the first layer
        assert self.layers[0].size == len(self.normalized_input), "Input size does not match the number of input nodes"
        self.layers[0].set_values(self.normalized_input)
        # Start propagating the input data through the network
        output = self.layers[0].feedforward()
        return output

    # Update the states of the nodes in the network by one timestep
    cpdef list back_propagation(self):
        cdef list output = []
        # Feed the input data into the first layer
        assert self.layers[-1].size == len(self.normalized_expected_output), "Output size does not match the number of output nodes"
        self.layers[-1].set_values(self.normalized_expected_output)
        # Start propagating the input data through the network
        output = self.layers[-1].feedbackward()
        return output

    # Normalize the input data to the range [-1, 1]
    cpdef void normalize_data(self, double[:] input_data):
        cdef double max_val = max(input_data)
        cdef double min_val = min(input_data)
        # Only need to normalize if the input data is not already in the range [-1, 1]
        if max_val == 1.0 and min_val == -1.0:
            self.normalized_input = input_data
            return
        cdef double[:] normalized = input_data
        for i in range(len(normalized)):
            normalized[i] = 2.0 * (normalized[i] - min_val) / (max_val - min_val) - 1.0
        self.normalized_input = normalized
    
    # Load a new normalized input frame into the network
    cpdef void load_data(self, double[:] input_data, double[:] expected_output):
        self.normalize_data(input_data)
        self.normalized_expected_output = expected_output

    # Reset the states of the nodes in the network
    cpdef void reset(self):
        for layer in self.layers:
            layer.reset()
        self.normalized_input = None
        self.normalized_expected_output = None

    # Perform inference on the input data
    cpdef tuple inference(self, double[:] input_data, double[:] expected_output):
        cdef list forward_output = []
        cdef list backward_output = []
        self.load_data(input_data, expected_output)
        forward_output = self.forward_propagation()
        backward_output = self.back_propagation()
        # Reduce change rates
        global CHANGE_RATE
        CHANGE_RATE *= 0.999
        self.reset()
        return (forward_output, backward_output)

cdef class Layer:
    cdef public int size
    cdef public set[Node] nodes
    cdef public Layer nextLayer
    cdef public Layer prevLayer
    cdef public str activation_type
    
    def __cinit__(self, int num_nodes, str activation_type = "BINARY", Layer nextLayer = None, Layer prevLayer = None):
        self.nodes = set()
        self.size = num_nodes
        self.nextLayer = nextLayer
        self.prevLayer = prevLayer
        self.activation_type = activation_type
        self.create_nodes(num_nodes)

    # Create the nodes in the network
    cpdef void create_nodes(self, int num_nodes):
        for i in range(num_nodes):
            self.nodes.add(Node(self.activation_type))

    cpdef void link(self, Layer target):
        self.nextLayer = target
        for node in self.nodes:
            for target_node in target.nodes:
                node.link(target_node)

    cpdef void set_values(self, double[:] values):
        for i, node in enumerate(self.nodes):
            node.nextState = values[i]

    cpdef list feedforward(self):
        cdef list layer_output = []
        for node in self.nodes:
            node.feedforward()
            layer_output.append(node.state)
        output = self.nextLayer.feedforward() if self.nextLayer is not None else layer_output
        return output

    cpdef list feedbackward(self):
        cdef list layer_output = []
        for node in self.nodes:
            node.feedbackward()
            layer_output.append(node.state)
        output = self.prevLayer.feedbackward() if self.prevLayer is not None else layer_output
        return output

    cpdef void reset(self):
        for node in self.nodes:
            node.reset()

cdef class Node:
    cdef public bint activated
    cdef public double state
    cdef public double nextState
    cdef public set[Edge] outputs
    cdef public set[Edge] inputs
    cdef public double bias
    cdef public str activation_type
    
    def __cinit__(self, str activation_type = "BINARY", double bias = 999):
        self.activated = False
        self.state = 0.0
        self.nextState = 0.0
        self.bias = self.randBias() if bias == 999 else bias
        self.outputs = set()
        self.inputs = set()
        self.activation_type = activation_type

    # Link the node to its output edges
    cpdef void link(self, Node target):
        cdef Edge edge = Edge(self, target)
        self.outputs.add(edge)
        target.inputs.add(edge)

    # Activation function
    cpdef void activation(self):
        self.activated = self.state > 0.0
        if self.activation_type == "BINARY":
            self.state = 0.0 if self.state < 0.0 else 1.0
        elif self.activation_type == "SIGMOID":
            self.state = 1.0 / (1.0 + exp(-self.state))
        elif self.activation_type == "RELU":
            self.state = 0.0 if self.state < 0.0 else self.state
        else:
            raise ValueError("Invalid activation function")

    # Update the state of the node by one timestep and feed the output to the connected edges
    cpdef void feedforward(self):
        self.state = self.nextState + self.bias
        self.nextState = 0.0
        self.activation()
        # if self.activated:
        #     for edge in self.inputs:
        #         if edge.source.activated:
        #             edge.increase_weight()
        # else:
        #     for edge in self.inputs:
        #         if edge.source.activated:
        #             edge.decrease_weight()
        for edge in self.outputs:
            edge.feedforward()

    cpdef void feedbackward(self):
        self.state = self.nextState + self.bias
        self.nextState = 0.0
        self.activation()
        if self.activated:
            for edge in self.outputs:
                if edge.target.activated:
                    edge.increase_weight()
                edge.target.reset()
        else:
            for edge in self.outputs:
                if edge.target.activated:
                    edge.decrease_weight()
                edge.target.reset()
        for edge in self.inputs:
            edge.feedbackward()

    # Reset the state of the node
    cpdef void reset(self):
        self.activated = False
        self.state = 0.0
        self.nextState = 0.0

    # Generate a random bias value for the node at initialization
    cpdef double randBias(self):
        cdef double random_number
        random_number = 2.0 * random.random() - 1.0
        return random_number

    cpdef void increase_bias(self):
        self.bias += CHANGE_RATE

    cpdef void decrease_bias(self):
        self.bias -= CHANGE_RATE

cdef class Edge:
    cdef public double weight
    cdef public Node source
    cdef public Node target
    
    def __cinit__(self, Node source, Node target, double weight = 999):
        self.weight = self.randWeight() if weight == 999 else weight
        self.source = source
        self.target = target

    # Feed the output of the source node to the target node
    cpdef void feedforward(self):
        cdef double output = self.source.state * self.weight
        self.target.nextState += output
    
    # Feed the output of the target node to the source node
    cpdef void feedbackward(self):
        cdef double output = self.target.state * self.weight
        self.source.nextState += output

    cpdef void increase_weight(self):
        self.weight += CHANGE_RATE

    cpdef void decrease_weight(self):
        self.weight -= CHANGE_RATE

    # Generate a random weight value for the edge at initialization
    cpdef double randWeight(self):
        cdef double random_number
        random_number = 2.0 * random.random() - 1.0
        return random_number
