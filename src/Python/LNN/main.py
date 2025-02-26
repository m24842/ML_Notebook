import sys
sys.path.append('src/Cython/LiquidNN')
from LNN import LiquidNeuralNetwork
from interpreter import Interpreter
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
    model = LiquidNeuralNetwork(9)
    interpreter = Interpreter()
    labels, pixels = import_data('data/emnist-mnist-train.csv')
    for label, pixel in zip(labels, pixels):
        liquid_state = model.inference(pixel)
        output = interpreter.train(label, liquid_state)
        model.reset()
        print(label, liquid_state, output, end="\r")