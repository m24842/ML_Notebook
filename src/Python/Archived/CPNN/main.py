import sys
sys.path.append('src/Cython/CriticalPhaseNN')
from CPNN import CriticalPhaseNeuralNetwork
import pandas as pd
import numpy as np
import pygame
import random
import cv2
import os

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

def display_net(screen, state, show_hidden=False):
    screen.fill((0, 0, 0))
    colors = {-1: (255, 0, 0), 0: (0, 0, 0), 1: (0, 255, 0)} if show_hidden else {-1: (0, 0, 0), 0: (0, 0, 0), 1: (255, 255, 255)}
    sorted_states = list(sorted(state.items(), key=lambda x: x[0][2], reverse=True)) if len(list(state.keys())[0]) > 2 else state.items()
    input_nodes = []
    for coords, node_info in sorted_states:
        if node_info[0] == 0 and not node_info[1]: continue
        color = colors.get(node_info[0])
        depth = 0
        if len(coords) > 2:
            depth = coords[2]
            color = tuple(int(round(i / (depth + 1))) for i in color)
            coords = coords[:2]
        pygame.draw.rect(screen, color, (PIXEL_SIZE*coords[1], PIXEL_SIZE*coords[0], PIXEL_SIZE, PIXEL_SIZE))
        if node_info[1]:
            input_nodes.append((coords, depth))
    # Draw input node outlines
    for coords, depth in input_nodes:
        color = (120, 170, 255)
        color = tuple(int(round(i / (depth + 1))) for i in color)
        pygame.draw.rect(screen, color, (PIXEL_SIZE*coords[1], PIXEL_SIZE*coords[0], PIXEL_SIZE, PIXEL_SIZE), 1)
            
def play_net_development(width, states_and_outputs, show_hidden=False):
    pygame.init()
    screen = pygame.display.set_mode((PIXEL_SIZE*width, PIXEL_SIZE*width))
    video_writer = cv2.VideoWriter('assets/playback.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(PIXEL_SIZE * width), int(PIXEL_SIZE * width)))
    clock = pygame.time.Clock()
    for i, (state, _) in enumerate(states_and_outputs):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        print(f"State: {i + 1}/{len(states_and_outputs)}", end='\r')
        display_net(screen, state, show_hidden)
        pygame.display.flip()
        
        # Capture video
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.flipud(np.rot90(frame))
        frame_surf = pygame.surfarray.make_surface(frame)
        pygame.image.save(frame_surf, 'assets/frame.png')
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        clock.tick(60)
    video_writer.release()
    os.remove('assets/frame.png')
    pygame.quit()

if __name__ == '__main__':
    dimensions = 3
    width = 50
    PIXEL_SIZE = 700 / width
    input_nodes = 784#int(width**dimensions * 0.01)
    output_nodes = 0
    epochs = 500
    new_model = input("Generate New Model? (y/n): ").lower() == 'y'
    if new_model: model = CriticalPhaseNeuralNetwork(
        width=width,
        dimensions=dimensions,
        num_input_nodes=input_nodes,
        connection_prob=0.24881,
    )
    else: model = CriticalPhaseNeuralNetwork(model_path = 'src/Saved_Models/model.npz')
    # inputs = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]] + [[0]*input_nodes] * (epochs - 3)
    # inputs = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]] + [[0]*input_nodes] * (epochs - 2)
    # inputs = [[0, 1, 0, 0, 0]] + [[0]*input_nodes] * (epochs - 1)
    # inputs = [[1 if idx == random.randint(0, input_nodes - 1) else 0 for idx in range(input_nodes)]] + [[0]*input_nodes] * (epochs - 1)
    inputs = [[1 if idx == random.randint(0, input_nodes - 1) else 0 for idx in range(input_nodes)] for _ in range(epochs)]
    # inputs = [[1] * input_nodes] * epochs
    inputs = [np.array(input, dtype=np.int32) for input in inputs]
    total_outputs = []
    for epoch, input  in zip(range(epochs), inputs):
        print(f"Epoch {epoch + 1}/{epochs}", end='\r')
        states_and_outputs = model.inference(input, max_time_steps = 1, crop = False)
        # for state, output in states_and_outputs: print(np.array(state), np.array(output))
        # if i >= 100: play_net_development(width, height, states_and_outputs)
        # play_net_development(width, height, states_and_outputs, True)
        total_outputs.extend(states_and_outputs)
    model.save_model('src/Saved_Models/model.npz')
    play_net_development(width, total_outputs, False)
    # print()
    # print(model.count_edges())
    # labels, pixels_data = import_data('data/emnist-mnist-train.csv')
    # total_correct = 0
    # for i, (label, pixels) in enumerate(zip(labels, pixels_data)):
    #     print(f"{i}/{len(labels)}", end="\r")
    #     expected_output = np.zeros((10,), dtype=np.double)
    #     expected_output[int(label)] = 1
    #     forward_output, backward_output = model.inference(pixels, expected_output)
    #     inference = np.argmax(forward_output)
    #     # print(label, inference, end='\n')
    #     total_correct += 1 if label == inference else 0
    # print(f"Accuracy: {100* total_correct / len(labels)}%")