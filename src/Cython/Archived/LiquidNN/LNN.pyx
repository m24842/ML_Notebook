import random
from libc.math cimport exp

cdef class LiquidNeuralNetwork:
    cdef public set[Node] nodes
    cdef public double[:] normalized_input
    cdef public int input_index
    
    def __cinit__(self, int num_nodes):
        self.nodes = set()
        self.create_nodes(num_nodes)
        self.input_index = 0

    # Create the nodes in the network
    cpdef void create_nodes(self, int num_nodes):
        for i in range(num_nodes):
            self.nodes.add(Node())
        for node in self.nodes:
            for other_node in self.nodes:
                # TBD: self-connection?
                if node != other_node:
                    node.link(other_node)
    
    # Update the states of the nodes in the network by one timestep
    cpdef list forward_propagation(self):
        # Get the next bit of input data to process
        cdef double input_data = self.normalized_input[self.input_index] if self.input_index < len(self.normalized_input) else 0.0
        cdef list output = []
        # Feed the input data into the nodes and update their states
        for node in self.nodes:
            node.nextState += input_data
            node.feedforward()
            # Record the node state after timestep
            output.append(node.state)
        return output

    # Normalize the input data to the range [-1, 1]
    cpdef void normalize_input(self, double[:] input_data):
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
    cpdef void load_input(self, double[:] input_data):
        self.normalize_input(input_data)
        self.input_index = 0

    # Reset the states of the nodes in the network
    cpdef void reset(self):
        for node in self.nodes:
            node.reset()
        self.input_index = 0
        self.normalized_input = None

    # Perform inference on the input data
    cpdef list inference(self, double[:] input_data):
        cdef list output = []
        self.load_input(input_data)
        while self.input_index < len(self.normalized_input):
            output = self.forward_propagation()
            self.input_index += 1
        return output

cdef class Node:
    cdef public double state
    cdef public double nextState
    cdef public set[Edge] outputs
    cdef public double bias
    
    def __cinit__(self, double bias = 999):
        self.state = 0.0
        self.nextState = 0.0
        self.bias = self.randBias() if bias == 999 else bias
        self.outputs = set()

    # Link the node to its output edges
    cpdef void link(self, Node target):
        self.outputs.add(Edge(self, target))

    # RELU activation function
    cpdef void activation(self):
        self.state = 0.0 if self.state < 0.0 else 1.0
        # self.state = 1.0 / (1.0 + exp(-self.state))

    # Update the state of the node by one timestep and feed the output to the connected edges
    cpdef void feedforward(self):
        self.state = self.nextState + self.bias
        self.nextState = 0.0
        self.activation()
        for edge in self.outputs:
            edge.feedforward()

    # Reset the state of the node
    cpdef void reset(self):
        self.state = 0.0
        self.nextState = 0.0

    # Generate a random bias value for the node at initialization
    cpdef double randBias(self):
        cdef double random_number
        random_number = 2.0 * random.random() - 1.0
        return random_number

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

    # Generate a random weight value for the edge at initialization
    cpdef double randWeight(self):
        cdef double random_number
        random_number = 2.0 * random.random() - 1.0
        return random_number
