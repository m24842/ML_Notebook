import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calc_correlation(prev_state, current_state):
    if current_state == 0:
        return 0
    return prev_state / current_state

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

class Network:
    def __init__(self):
        self.layers = [
            Layer(784, 10, "relu"),
            # Layer(36, 10, "tanh"),
            # Layer(36, 10, "relu"),
        ]
    
    def forward(self, inputs, visualize=False):
        for layer in self.layers:
            if visualize: plot_net_data(inputs.reshape(inputs.shape[0], 1), 'Forward Pass')
            inputs = layer.forward(inputs)
        if visualize: plot_net_data(inputs.reshape(inputs.shape[0], 1), 'Forward Pass')
        # inputs = softmax(inputs)
        return inputs
    
    def backward(self, input, expected_output):
        new_parameters = []
        self.layers[-1].state = expected_output
        for i in range(len(self.layers) - 1, 0, -1):
            new_parameters.append(self.layers[i].backward(self.layers[i - 1].state))
        new_parameters.append(self.layers[0].backward(input))
        new_parameters.reverse()
        return new_parameters

    def update_parameters(self, new_parameters):
        new_param_influence = 0.5
        old_param_influence = 1 - new_param_influence
        for i in range(len(self.layers)):
            self.layers[i].biases = self.layers[i].biases * old_param_influence + new_parameters[i][0] * new_param_influence
            self.layers[i].weights = self.layers[i].weights * old_param_influence + new_parameters[i][1] * new_param_influence
    
    def clear_states(self):
        for layer in self.layers:
            layer.state = np.zeros(layer.output_size)

class Layer:
    def __init__(self, input_size, output_size, activation="relu"):
        self.output_size = output_size
        self.input_size = input_size
        self.weights = np.random.uniform(-1, 1, (self.input_size, self.output_size))#np.ones((self.input_size, self.output_size))#
        self.biases = np.random.uniform(-1, 1, self.output_size)#np.ones(self.output_size)#
        self.state = np.zeros(self.output_size)
        self.activation = activation
        
    def activate(self):
        if self.activation == "relu":
            self.state = relu(self.state)
        elif self.activation == "sigmoid":
            self.state = sigmoid(self.state)
        elif self.activation == "tanh":
            self.state = tanh(self.state)
        else:
            return

    def forward(self, inputs):
        assert inputs.shape[0] == self.input_size
        self.state = np.dot(inputs, self.weights) + self.biases
        self.activate()
        return self.state
    
    def backward(self, prev_layer_state):
        bias_correlation = np.zeros(self.output_size)
        prev_layer_correlation = np.zeros((self.input_size, self.output_size))
        for i in range(self.output_size):
            current_state = self.state[i]
            # Calculate bias correlation
            bias_correlation[i] = calc_correlation(1, current_state)
            for j in range(self.input_size):
                prev_state = prev_layer_state[j]
                prev_layer_correlation[j, i] = calc_correlation(prev_state, current_state)
                
        return bias_correlation, prev_layer_correlation

def plot_net_data(weights, title):
    num_output_nodes = weights.shape[1]
    
    # Calculate the number of rows and columns
    cols = int(np.ceil(np.sqrt(num_output_nodes)))  # Set the number of columns to be the square root of the number of output nodes
    rows = int(np.ceil(num_output_nodes / cols))  # Calculate the required number of rows

    # Create a subplot with the calculated number of rows and columns
    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
    
    # Flatten the axes array for easy indexing
    axes = np.array(axes).flatten()
    
    # Get the dimensions for reshaping
    height, width = int(np.sqrt(weights.shape[0])), int(np.sqrt(weights.shape[0]))
    if height * width != weights.shape[0]:  # If not a perfect square, calculate dimensions differently
        width = weights.shape[0]
        height = 1
    
    norm = plt.Normalize(weights.min(), weights.max())
    
    for i in range(num_output_nodes):
        ax = axes[i]
        weight_image = weights[:, i].reshape(height, width).T  # Reshape to the calculated dimensions and transpose
        ax.imshow(weight_image, cmap='viridis', aspect='auto', norm=norm)
        if height * width < 100:
            for row in range(height):
                for col in range(width):
                    rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(rect)
        ax.set_title(f'{i}', fontsize=5)
        ax.axis('off')
    
    # Turn off any unused subplots
    for j in range(num_output_nodes, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(title, fontsize=5)
    plt.tight_layout()
    plt.show()
    
def read_data():
    data = pd.read_csv('/Users/matthewhua/Downloads/archive/emnist-mnist-train.csv', header=None)
    data = data.sample(frac=1).reset_index(drop=True)

    # Extract labels and pixel values
    labels = data.iloc[:, 0].values
    pixels = data.iloc[:, 1:].values
    expected_outputs = one_hot_encode(labels, 10)
    inputs = np.array(pixels, dtype=np.float32)
    
    return inputs, expected_outputs

def main():
    net = Network()
    # inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 
    # expected_outputs = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]) # 
    for batch in range(100):
        inputs, expected_outputs = read_data()
        correlations = []
        accuracy = 0
        print(f'Batch: {batch + 1}')
        amount = 1000
        visualize = False#input('Visualize? (y/n): ') == 'y'
        # amount = ""
        # while not amount.isdigit(): amount = input('Enter amount of data: ')
        # amount = int(amount)
        # amount = np.min([amount, len(inputs)])
        for input_data, expected_output in zip(inputs[:amount], expected_outputs[:amount]):
            inference = net.forward(input_data, visualize)
            loss = np.where(np.argsort(inference)[::-1] == np.argmax(expected_output))[0][0]
            if loss == 0: accuracy += 1
            # print(np.argmax(expected_output), loss)
            new_parameters = net.backward(input_data, expected_output)
            correlations.append(new_parameters)
            # net.update_parameters(new_parameters)
        mean_biases = [np.mean([corr[i][0] for corr in correlations], axis=0) for i in range(len(net.layers))]
        mean_weights = [np.mean([corr[i][1] for corr in correlations], axis=0) for i in range(len(net.layers))]
        new_parameters = [(mean_biases[i], mean_weights[i]) for i in range(len(net.layers))]
        net.update_parameters(new_parameters)
        print(f'Accuracy: {100 * accuracy / amount} %')
        for i, layer in enumerate(net.layers):
            plot_net_data(layer.weights, f'Layer {i + 1} Weights After Batch {batch+1}')
            # plot_net_data(np.array([layer.biases]), f'Layer {i + 1} Biases After Batch {batch+1}')

if __name__ == "__main__":
    main()