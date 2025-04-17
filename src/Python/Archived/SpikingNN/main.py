import sys
sys.path.append('src/Python')
sys.path.append('src/Cython')
from SpikingNN.SNN import SpikingNN
from Tools.eventCam import EventBasedCamera
from Games.pong import PongGame
import numpy as np
import time

def printOutput(output):
    outputStr = ['█' if value == 1 else ' ' for value in output]
    print(f"|{''.join(outputStr)}|")

def oneHot(size, index):
    oneHot = np.zeros(size).tolist()
    oneHot[index] = 1
    return oneHot

def printIOE(input, output, expected):
    for i in range(len(input)):
        iv = ['█' if value == 1 else ' ' for value in input[i]]
        ov = ['█' if value == 1 else ' ' for value in output[i]]
        ev = ['█' if value == 1 else ' ' for value in expected[i]]
        iv = ''.join(iv)
        ov = ''.join(ov)
        ev = ''.join(ev)
        print(f"{iv} | {ov} | {ev}")

if __name__ == '__main__':
    netSize = 500
    inputSize = 400
    outputSize = 2
    
    snn = SpikingNN(
        # seed=np.random.random()*10000,
        seed=1,
        numNodes=netSize,
        inputSize=inputSize,
        outputSize=outputSize,
        decayRate=0.9,
    )
    
    # Load saved model
    try:
        snn.load("src/Saved_Models/snn_model.npz")
        print("Model loaded successfully!")
    except:
        print("Model not found!")
        pass
    
    # Generate synthetic data for supervised learning
    rounds = 2
    roundLength = 100
    inputs = []
    frames = []
    for _ in range(rounds):
        env = PongGame(1 if _ % 2 == 0 else -1)
        vision = EventBasedCamera(threshold=0, resolution=(np.sqrt(inputSize), np.sqrt(inputSize)))
        roundInputs = []
        roundFrames = np.zeros((roundLength, inputSize), dtype=np.int32)
        for _ in range(roundLength):
            ballPos = env.getBallPosition()
            paddlePos = env.getPaddlePosition()
            command = "LEFT" if paddlePos[0] > ballPos[0] else "RIGHT" if paddlePos[0] < ballPos[0] else ""
            roundInputs.append([1, 0] if command == "LEFT" else [0, 1] if command == "RIGHT" else [0, 0])
            env.loadInput(command)
            env.step()
            frame = vision.processFrame(env.getScreen()).T
            vision.showFrame(frame)
            roundFrames[_] = frame.flatten()
        inputs.append(roundInputs)
        frames.append(roundFrames)
        env.quit()
        vision.quit()
    
    samples = rounds
    rawInput = np.array(frames, dtype=np.int32)
    targetOutput = np.array(inputs, dtype=np.float32)
    
    np.savez("data/snn_data.npz", samples=samples, rawInput=rawInput, targetOutput=targetOutput)
    
    # Load synthetic data
    length = 1000
    data = np.load("data/snn_data.npz")
    samples = 1#data['samples']
    rawInput = data['rawInput'][:samples,:length]
    targetOutput = data['targetOutput'][:samples,:length]
    
    # Run a game
    env = PongGame(1)
    vision = EventBasedCamera(threshold=0, resolution=(np.sqrt(inputSize), np.sqrt(inputSize)))
    env.start()
    inputs = []
    frames = []
    while env.isRunning():
        time.sleep(1/60)
        inputs.append(env.readInput())
        frame = vision.processFrame(env.getScreen()).T
        frames.append(frame)
        vision.showFrame(frame)
    env.quit()
    vision.quit()
    
    # samples = 1
    # frames = [frame.flatten() for frame in frames]
    # rawInput = np.array([frames], dtype=np.int32)
    # targetOutput = np.array([inputs], dtype=np.float32)
    
    # np.random.seed(10)
    # samples = 1
    # inputLength = 1000
    # rawInput = [[np.random.choice([0, 1], inputSize, p=[0.6, 0.4]) for _ in range(inputLength)] for _ in range(samples)]
    # # rawInput = np.array([[dataRange] * inputSize], dtype=np.int32)
    # rawInput = np.array(rawInput, dtype=np.int32)
    # targetOutput = [[np.random.choice([0, 1], outputSize, p=[0.6, 0.4]) for _ in range(inputLength)] for _ in range(samples)]
    # targetOutput = np.array(targetOutput, dtype=np.float32)
    # for i in range(samples):
    #     if sum(sum(rawInput[i])) == 0: targetOutput[i] = np.array([[0]] * inputLength, dtype=np.int32)
    #     targetOutput[i, :(np.argmax(rawInput[i]) + 1)] = np.array([0] * (np.argmax(rawInput[i]) + 1)).reshape(-1, 1)
    # # targetOutput = np.array([[1]] * samples, dtype=np.int32)
    
    # # printIOE(rawInput[0], targetOutput[0], targetOutput[0])

    # Supervised training
    best = np.inf
    avgError = np.inf
    sumError = np.inf
    count = 0
    for _ in range(100):
    # while avgError > 0:
        avgError = 0
        sumError = 0
        
        for i in range(samples):
            for j in range(1):
                loss, output, states = snn.train(rawInput[i], targetOutput[i], False)
            # for o in output:
            #     printOutput(o)
            # printIOE(rawInput[i], output, targetOutput[i])
            avgError += loss
            sumError += loss
            snn.reset()
            snn.save("src/Saved_Models/snn_model.npz")
            ravgError = avgError / (i+1)
            # print(f"Running Error: {ravgError:.3f}")
        avgError /= samples
        best = min(best, avgError)
        print(f"Epoch: {count} | Error: {avgError:.3f} | Best: {best:.3f}")
        print()
        count +=1
    
    # # Evaluate
    # best = np.inf
    # avgError = 0
    # sumError = 0
    # for i in range(samples):
    #     loss, output, states = snn.evaluate(rawInput[i], targetOutput[i], False)
        # for o in output:
        #     printOutput(o)
    #     # printIOE(rawInput[i], output, targetOutput[i])
    #     avgError += loss
    #     sumError += loss
    #     snn.reset()
    #     ravgError = avgError / (i+1)
    #     # print(f"Running Error: {ravgError:.3f}")
    # avgError /= samples
    # best = min(best, avgError)
    # print(f"Error: {avgError:.3f} | Best: {best:.3f}")
    
    # Run a game
    count = 0
    while True:
        count += 1
        env = PongGame(1, noEnd=False) # if count % 2 == 0 else -1
        vision = EventBasedCamera(threshold=0, resolution=(np.sqrt(inputSize), np.sqrt(inputSize)))
        # env.start()
        env.setRunning(True)
        while env.isRunning():
            # time.sleep(1/60)
            ballPos = env.getBallPosition()
            paddlePos = env.getPaddlePosition()
            targetOutput = [1, 0] if paddlePos[0] > ballPos[0] else [0, 1] if paddlePos[0] < ballPos[0] else [0, 0]
            targetOutput = np.array(targetOutput, dtype=np.float32)
            frame = vision.processFrame(env.getScreen()).T
            vision.showFrame(frame)
            frame = np.array(frame, dtype=np.int32).flatten()
            for _ in range(1):
                output = snn.inference(frame, None, False).astype(np.int32)
            command = "LEFT" if output[0] == 1 and output[1] == 0 else "RIGHT" if output[1] == 1 and output[0] == 0 else ""
            env.loadInput(command)
            env.step()
            # print(f"Output: {output} | Target: {targetOutput}")
        env.quit()
        vision.quit()
        snn.reset()
        snn.save("src/Saved_Models/snn_model.npz")