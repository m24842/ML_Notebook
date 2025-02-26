import sys
sys.path.append('src/Cython/HebbianNN')
from HNN import HebbianNeuralNetwork
import pandas as pd
import numpy as np

def import_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path, header=None, dtype=np.double)

    # Extract labels and pixel values
    labels = data.iloc[:, 0].values
    pixels = data.iloc[:, 1:].values
    
    # Set pixel values to 1 if greater than zero, else set to 0
    pixels[pixels > 0] = 1
    pixels[pixels <= 0] = 0

    return labels, pixels

if __name__ == '__main__':
    shape = np.array([784, 10, 10], dtype=np.int32)
    activations = ["RELU", "RELU", "SIGMOID"]
    model = HebbianNeuralNetwork(shape, activations)
    labels, pixels_data = import_data('data/emnist-mnist-train.csv')
    total_correct = 0
    for i, (label, pixels) in enumerate(zip(labels, pixels_data)):
        print(f"{i}/{len(labels)}", end="\r")
        expected_output = np.zeros((10,), dtype=np.double)
        expected_output[int(label)] = 1
        forward_output, backward_output = model.inference(pixels, expected_output)
        inference = np.argmax(forward_output)
        # print(label, inference, end='\n')
        total_correct += 1 if label == inference else 0
    print(f"Accuracy: {100* total_correct / len(labels)}%")