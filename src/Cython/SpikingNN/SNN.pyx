import numpy as np
cimport numpy as cnp
import matplotlib.pyplot as plt

cdef class SpikingNN:
    cdef public int numNodes
    cdef public object inputIndices
    cdef public object outputIndices
    cdef public object connections
    cdef public object states
    cdef public object potentials
    cdef public object activities
    cdef public float decayRate
    cdef public object targetActivity
    
    def __cinit__(self, int seed=0, int numNodes=10, int inputSize=1, int outputSize=1, float decayRate=0.5):
        np.random.seed(seed)
        self.numNodes = numNodes

        # Create random connections
        assert inputSize + outputSize <= numNodes, "Invalid input/output size"
        indices = np.random.choice(numNodes, size=inputSize+outputSize, replace=False)
        self.inputIndices = indices[:inputSize]
        self.outputIndices = indices[inputSize:]
        self.connections = self.connect(numNodes)
        self.states = np.zeros(numNodes, dtype=np.float32)
        self.potentials = np.full(numNodes, 0, dtype=np.float32)
        self.decayRate = decayRate
        self.activities = np.zeros(numNodes, dtype=np.float32)
        self.targetActivity = np.zeros(numNodes, dtype=np.float32)

    cpdef connect(self, numNodes):
        connections = 1e-2 * np.random.uniform(-1, 1, size=(numNodes, numNodes)).astype(np.float32)
        # np.fill_diagonal(connections, 0)
        return connections

    cpdef forward(self, object sample=None, object target=None, visualize=False):
        if sample is None: sample = np.zeros(len(self.inputIndices), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] changes = self.states @ self.connections.T
        self.potentials += changes.T

        cdef cnp.ndarray[cnp.npy_bool, ndim=1] spikes = self.potentials >= 1
        spikes[self.inputIndices] = sample.astype(np.bool_)

        self.states = np.where(spikes, 1, 0).astype(np.float32)
        self.states[self.inputIndices] = sample
        
        self.potentials[spikes] = 0
        self.potentials *= self.decayRate

        self.update(target, visualize)

        self.activities = self.decayRate * self.activities + (1 - self.decayRate) * self.states

    cpdef update(self, object target=None, visualize=False):
        if target is None: return
        cdef cnp.ndarray[cnp.float32_t, ndim=1] targets = np.abs(self.connections.T) @ self.activities - 0.5 / self.numNodes
        targets[self.inputIndices] = self.activities[self.inputIndices]
        targets[self.outputIndices] = target[self.outputIndices]
        np.clip(targets, -1, 1, out=targets)
        targets[targets < 0] = 1 - targets[targets < 0]
        cdef cnp.ndarray[cnp.float32_t, ndim=1] errors = targets - self.activities
        self.connections += 1e-2 * np.outer(errors, self.activities)
        # np.fill_diagonal(self.connections, 0)

        if visualize: self.plot(targets)

    cpdef plot(self, object targets):
        fig = plt.gcf()
        plt.style.use('dark_background')
        fig.canvas.manager.set_window_title('SNN Data')
        if len(fig.get_axes()) < 2:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.set_size_inches(18.5, 10.5)
            # fig.tight_layout()
        else:
            ax1, ax2 = fig.get_axes()
        ax1.clear()
        ax2.clear()

        for idx in self.inputIndices:
            ax1.axvline(x=idx, color='gold', linestyle='-', label='input' if idx == self.inputIndices[0] else "")
        for idx in self.outputIndices:
            ax1.axvline(x=idx, color='white', linestyle='-', label='output' if idx == self.outputIndices[0] else "", linewidth=0.5)
        ax1.plot(self.activities, color='red', label='activities')
        ax1.plot(targets, color='blue', linestyle=':', label='targets')
        ax1.set_ylim(0, 1)
        if not ax1.get_legend():
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)

        ax2.imshow(self.connections.T, cmap='viridis')
        maxWeight = np.max(self.connections)
        minWeight = np.min(self.connections)
        ax2.set_title(f"Min: {minWeight:.3f} Max: {maxWeight:.3f}")

        plt.draw()
        plt.pause(1/100)

    cpdef object inference(self, cnp.ndarray[cnp.int32_t, ndim=1] inputVector, object targetVector=None, visualize=False):
        if targetVector is not None:
            self.targetActivity[self.outputIndices] = self.decayRate * self.targetActivity[self.outputIndices] + (1 - self.decayRate) * targetVector #
            self.forward(inputVector, self.targetActivity, visualize)
        else:
            self.forward(inputVector, None, visualize)
        return self.getOutput()

    cpdef object inferenceSeries(self, cnp.ndarray[cnp.int32_t, ndim=2] inputMatrix, object targetMatrix=None, visualize=False):
        samples = inputMatrix.shape[0]
        cdef cnp.ndarray[cnp.float32_t, ndim=2] outputs = np.zeros((samples, self.outputIndices.shape[0]), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] states = np.zeros((samples, self.numNodes), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] activities = np.zeros((samples, self.outputIndices.shape[0]), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] targetActivities = np.zeros((samples, self.numNodes), dtype=np.float32)
        cdef cnp.int32_t i

        if targetMatrix is not None:
            for i, target in enumerate(targetMatrix):
                self.targetActivity[self.outputIndices] = self.decayRate * self.targetActivity[self.outputIndices] + (1 - self.decayRate) * targetMatrix[i] #
                targetActivities[i] = self.targetActivity.copy()

        for i, sample in enumerate(inputMatrix):
            self.forward(sample, None if targetMatrix is None else targetActivities[i], visualize)
            outputs[i] = self.getOutput()
            states[i] = self.states
            activities[i] = self.activities[self.outputIndices]
            self.states[self.outputIndices] = targetMatrix[i]
        return outputs, states, activities

    cpdef cnp.float32_t loss(self, object a, object b):
        mseLoss = np.sum((a - b) ** 2)
        return mseLoss
    
    cpdef object error(self, object actual, object expected):
        return actual - expected

    cpdef object train(self, cnp.ndarray[cnp.int32_t, ndim=2] inputMatrix, cnp.ndarray[cnp.float32_t, ndim=2] targetMatrix, visualize=False):
        assert targetMatrix.shape[1] == self.outputIndices.shape[0], "Invalid vector shapes"
        
        samples = inputMatrix.shape[0]
        cdef cnp.ndarray[cnp.float32_t, ndim=2] targetActivity = np.zeros((samples, len(self.outputIndices)), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] tempActivity = np.zeros(len(self.outputIndices), dtype=np.float32)
        if targetMatrix is not None:
            for i, target in enumerate(targetMatrix):
                tempActivity = self.decayRate * tempActivity + (1 - self.decayRate) * targetMatrix[i]
                targetActivity[i] = tempActivity.copy()
        
        outputs, states, activities = self.inferenceSeries(inputMatrix, targetMatrix, visualize)
        loss = self.loss(activities, targetActivity)
        return loss, outputs, states

    cpdef object evaluate(self, cnp.ndarray[cnp.int32_t, ndim=2] inputMatrix, cnp.ndarray[cnp.float32_t, ndim=2] targetMatrix, visualize=False):
        outputs, states, activities = self.inferenceSeries(inputMatrix, targetMatrix, visualize)
        loss = self.loss(activities, targetMatrix)
        return loss, outputs, states

    cpdef object getOutput(self):
        return self.states[self.outputIndices]

    cpdef reset(self):
        self.states = np.zeros(self.numNodes, dtype=np.float32)
        self.potentials = np.zeros(self.numNodes, dtype=np.float32)
        self.activities = np.zeros(self.numNodes, dtype=np.float32)
        self.targetActivity = np.zeros(self.numNodes, dtype=np.float32)

    cpdef save(self, str filename):
        np.savez(filename, numNodes=self.numNodes, inputIndices=self.inputIndices, outputIndices=self.outputIndices, connections=self.connections, decayRate=self.decayRate)
    
    cpdef load(self, str filename):
        data = np.load(filename)
        self.numNodes = data['numNodes']
        self.inputIndices = data['inputIndices']
        self.outputIndices = data['outputIndices']
        self.connections = data['connections']
        self.decayRate = data['decayRate']
        self.states = np.zeros(self.numNodes, dtype=np.float32)
        self.potentials = np.zeros(self.numNodes, dtype=np.float32)
        self.activities = np.zeros(self.numNodes, dtype=np.float32)
        self.targetActivity = np.zeros(self.numNodes, dtype=np.float32)