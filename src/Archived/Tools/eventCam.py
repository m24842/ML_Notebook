import cv2
import numpy as np

class EventBasedCamera:
    def __init__(self, threshold=10, resolution=(100, 100)):
        self.resolution = (int(resolution[0]), int(resolution[1]))  # Ensure resolution values are integers
        self.threshold = threshold
        self.prevFrame = np.zeros(self.resolution, dtype=np.uint8)
        self.eventFrame = np.zeros(self.resolution, dtype=np.uint8)

    def processFrame(self, currFrame):
        """Process the current frame and return an image with events highlighted."""
        # Resize the current frame to the specified resolution
        currFrameResized = cv2.resize(currFrame, self.resolution, interpolation=cv2.INTER_AREA)
        currFrameGray = cv2.cvtColor(currFrameResized, cv2.COLOR_BGR2GRAY)

        # Detect events
        currFrameGray[currFrameGray > 0] = 1
        self.detectEvents(currFrameGray)
        self.prevFrame = currFrameGray
        return self.eventFrame

    def detectEvents(self, currFrame):
        """Detect changes between two frames."""
        # Calculate the absolute difference
        diff = cv2.absdiff(self.prevFrame, currFrame)

        # Threshold the difference image
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        self.eventFrame = thresh
    
    def showFrame(self, useFrame=None):
        frame = self.prevFrame.T.copy() if useFrame is None else useFrame.copy()
        frame[frame > 0] = 255
        frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Vision', frame)
    
    def quit(self):
        cv2.destroyAllWindows()
