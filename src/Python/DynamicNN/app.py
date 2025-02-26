import sys
import time
sys.path.append('src/Python')
from DynamicNN.main import Net
from Tools.eventCam import EventBasedCamera
from Games.pong import PongGame
import cv2
import torch
import numpy as np

if __name__ == "__main__":
    io = 2502
    hidden = 100
    norm = 1
    processLength = 1
    net = Net(io, hidden, norm, processLength)
    net.loadExisting('src/Saved_Models/net.pth')
    
    resolution = 50
    
    # env = PongGame(1)
    # state = torch.zeros((io + hidden))  # Network state
    # move = [0, 0]
    # i = 0
    # while True:
    #     i += 1

    #     frame = env.getScreen()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    #     frame[frame > 0] = 1
    #     input_vector = torch.cat((torch.tensor(move), torch.tensor(frame).flatten(), state[io:]))
        
    #     # Forward pass
    #     out = net(input_vector.clone().detach())
    #     state = out.clone().detach()
        
    #     # Rule-based command as target
    #     ball_pos = env.getBallPosition()
    #     paddle_pos = env.getPaddlePosition()
    #     command = "LEFT" if ball_pos[0] < paddle_pos[0] else "RIGHT" if ball_pos[0] > paddle_pos[0] else ""
    #     move = [1, 0] if command == "LEFT" else [0, 1] if command == "RIGHT" else [0, 0]
    #     env.loadInput(command)
    #     env.step()

    #     # Target for prediction
    #     next_frame = env.getScreen()
    #     next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    #     next_frame = cv2.resize(next_frame, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    #     next_frame[next_frame > 0] = 1
    #     target = torch.cat((torch.tensor(move), torch.tensor(next_frame).flatten(), input_vector[io:]))
        
    #     # Compute loss and backpropagation
    #     loss = net.backward(out, target)
        
    #     if i % 100 == 0:
    #         i = 0
    #         net.save('src/Saved_Models/net.pth')
    # env.quit()
    
    # for _ in range(100):
    #     input_tensor = torch.tensor(np.array(inputs)).float()
    #     frame_tensor = torch.tensor(np.array(frames)).float()
    #     combined_tensor = torch.cat((input_tensor, frame_tensor), dim=1)
    #     outputs = net.forwardBatch(combined_tensor[:-1])
    #     targets = torch.cat((combined_tensor[1:, :io], outputs[:, io:]), dim=1)
    #     loss = net.backward(outputs, targets)
    #     print(loss)
    #     net.save('src/Saved_Models/net.pth')

    env = PongGame(1)
    env.setRunning(True)
    move = [0, 0]
    netIO = torch.zeros((io+hidden))
    while env.isRunning():
        time.sleep(1/60)
        frame = env.getScreen()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        frame[frame > 0] = 1
        netIO = torch.cat((torch.tensor(move), torch.tensor(frame).flatten(), netIO[io:]))
        netIO = net(netIO)
        move = netIO[:2].tolist()
        print(move)
        frame[frame > 0] = 255
        temp = cv2.resize(frame.T, (100, 100), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Frame", temp)
        move = [max(0, min(1, int(round(m)))) for m in move]
        command = "LEFT" if move[0] == 1 and move[1] == 0 else "RIGHT" if move[1] == 1 and move[0] == 0 else ""
        env.loadInput(command)
        env.step()
    env.quit()
    
    net.eval()
    netIO = torch.zeros((io+hidden))
    for _ in range(1000):
        netIO = net(netIO)
        netIO = torch.clamp(netIO, 0, 1)
        print(netIO)
        frame = netIO[2:io].clone().detach().numpy().reshape(resolution, resolution).T
        frame *= 255
        np.clip(frame, 0, 255, out=frame)
        frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Frame", frame)
        cv2.waitKey(int(1000/10))