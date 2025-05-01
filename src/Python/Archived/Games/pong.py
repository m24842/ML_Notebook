import pygame
from threading import Thread, Lock
from queue import Queue
import numpy as np

class PongGame:
    def __init__(self, startDirection=1, noEnd=False):
        pygame.init()
        
        self.noEnd = noEnd
        
        # Game variables
        self.WIDTH, self.HEIGHT = 800, 600
        self.BALL_RADIUS = 15
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 10
        ballSpeed = 5
        self.ballSpeedX, self.ballSpeedY = startDirection * ballSpeed, ballSpeed
        self.PADDLE_SPEED = 7
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        
        # Screen setup
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Pong')
        self.currentScreen = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        
        # Game objects
        self.ball = pygame.Rect(self.WIDTH // 2 - self.BALL_RADIUS, 0, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        self.paddle = pygame.Rect(self.WIDTH // 2 - self.PADDLE_WIDTH // 2, self.HEIGHT - self.PADDLE_HEIGHT - 10, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        self.score = 0
        self.clock = pygame.time.Clock()
        self.inputQueue = Queue()
        
        self.running = False  # Flag to indicate game is running
        self.lock = Lock()  # Lock for thread-safe access to screen
    
    def readInput(self):
        """Read physical inputs."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            self.quit()
            return [0, 0]
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.loadInput("LEFT")
            return [1, 0]
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.loadInput("RIGHT")
            return [0, 1]
        else:
            self.loadInput()
            return [0, 0]
        
    def loadInput(self, command=""):
        """Load input commands into the input queue: 'LEFT' or 'RIGHT'."""
        pygame.event.pump()
        self.inputQueue.put(command)

    def handleInput(self):
        """Handles paddle movement input asynchronously from a queue."""
        while not self.inputQueue.empty():
            command = self.inputQueue.get()
            if command == 'LEFT' and self.paddle.left > 0:
                self.paddle.x -= self.PADDLE_SPEED
            elif command == 'RIGHT' and self.paddle.right < self.WIDTH:
                self.paddle.x += self.PADDLE_SPEED

    def update(self):
        """Update the game state: ball movement and collisions."""
        # Ball movement
        self.ball.x += self.ballSpeedX
        self.ball.y += self.ballSpeedY

        # Ball collision with left and right walls
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ballSpeedX = -self.ballSpeedX

        # Ball out of bounds (game over)
        if not self.noEnd and self.ball.bottom >= self.HEIGHT:
            print(f'Game Over!\nFinal score: {self.score}')
            self.running = False
            return
        elif self.ball.bottom >= self.HEIGHT:
            self.ballSpeedY = -self.ballSpeedY

        # Ball collision with top wall
        if self.ball.top <= 0:
            self.ballSpeedY = -self.ballSpeedY

        # Ball collision with paddle
        if self.ball.colliderect(self.paddle):
            self.ballSpeedY = -self.ballSpeedY
            self.score += 1
            print(f'Score: {self.score}', end='\r')

    def render(self):
        """Draw the game objects on the screen."""
        self.screen.fill(self.BLACK)
        pygame.draw.ellipse(self.screen, self.WHITE, self.ball)
        pygame.draw.rect(self.screen, self.WHITE, self.paddle)

        # Update display
        pygame.display.flip()

        # Use a lock for thread-safe access to the screen data
        with self.lock:
            self.currentScreen = pygame.surfarray.array3d(self.screen)

    def getScreen(self):
        """Return the current screen as a numpy array (thread-safe)."""
        with self.lock:
            return np.copy(self.currentScreen)
    
    def getBallPosition(self):
        """Return the ball's position."""
        return self.ball.x + self.BALL_RADIUS, self.ball.y + self.BALL_RADIUS

    def getPaddlePosition(self):
        """Return the paddle's position."""
        return self.paddle.x + self.PADDLE_WIDTH / 2, self.paddle.y + self.PADDLE_HEIGHT / 2

    def gameLoop(self):
        """Run the game loop asynchronously, checking for input commands from the queue."""
        while self.running:
            # Handle input from the queue
            self.handleInput()

            # Update game state
            self.update()

            # Render game
            self.render()

            # Control frame rate
            self.clock.tick(60)
            
            if not self.running: break

    def start(self):
        """Start the game loop in a new thread."""
        gameThread = Thread(target=self.gameLoop)
        self.RUNNING = True
ENTITY = os.getenv("WANDB_API_KEY")
        gameThread.start()
        return gameThread
    
    def setRunning(self, running):
        """Set the running flag."""
        self.running = running

    def step(self):
        """Run a single step of the game loop."""
        self.handleInput()
        self.update()
        self.render()

    def isRunning(self):
        """Check if the game is still running."""
        return self.running

    def quit(self):
        """Quit the game."""
        self.running = False
        pygame.display.quit()