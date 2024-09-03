import pygame
import sys
import random
from pygame.math import Vector2

class SNAKE:
    def __init__(self):
        # Initialize the snake in a straight line with no overlaps
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.direction = Vector2(0, 0)  # Move to the right initially
        self.direction_queue = []  # Queue to store direction changes
        self.new_block = False  # Flag for adding new blocks

    def draw_snake(self):
        for block in self.body:
            xPos = int(block.x * cell_size)
            yPos = int(block.y * cell_size)
            block_rect = pygame.Rect(xPos, yPos, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 0, 255), block_rect)

    def move_snake(self):
        # Apply all valid direction changes from the queue
        if self.direction_queue:
            new_direction = self.direction_queue.pop(0)
            # Avoid reversing into itself
            if new_direction.x != -self.direction.x or new_direction.y != -self.direction.y:
                self.direction = new_direction

        # Update snake body
        if self.new_block:
            body_copy = self.body[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]

        body_copy.insert(0, body_copy[0] + self.direction)
        self.body = body_copy[:]

    def add_block(self):
        self.new_block = True

class FRUKT:
    def __init__(self):
        self.randomize()

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x * cell_size), int(self.pos.y * cell_size), cell_size, cell_size)
        pygame.draw.rect(screen, (200, 0, 0), fruit_rect)

    def randomize(self, snake=None):
        while True:
            self.x = random.randint(0, cell_count - 1)
            self.y = random.randint(0, cell_count - 1)
            self.pos = Vector2(self.x, self.y)
            
            # Check if the fruit's position is on the snake's body
            if snake is None or self.pos not in snake.body:
                break

def check_collision():
    global score  # Use the global score variable
    if snake.body[0] == fruit.pos:
        fruit.randomize(snake)  # Pass the snake object to check its body
        snake.add_block()
        score += 10  # Increase the score when snake eats a fruit

def check_fail():
    # Check if snake hits the wall
    if not 0 <= snake.body[0].x < cell_count or not 0 <= snake.body[0].y < cell_count:
        print("Snake hit the wall")
        return True
    # Check if snake hits itself
    for block in snake.body[1:]:
        if block == snake.body[0]:
            print("Snake hit itself")
            return True
    return False

def draw_score():
    score_surface = score_font.render(f'Score: {score}', True, (56, 74, 12))
    score_rect = score_surface.get_rect(center=(cell_size * 1.5, cell_size * 0.5))
    screen.blit(score_surface, score_rect)

def draw_background():
    for row in range(cell_count):
        for col in range(cell_count):
            if (row + col) % 2 == 0:
                color = (170, 215, 81)  # Light green color
            else:
                color = (162, 209, 73)  # Dark green color
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)

pygame.init()

cell_size = 40
cell_count = 15
screen = pygame.display.set_mode((cell_count * cell_size, cell_count * cell_size))
clock = pygame.time.Clock()

# Initialize the snake and fruit
snake = SNAKE()
fruit = FRUKT()
fruit.randomize(snake)  # Ensure the fruit spawns in a valid position

# Initialize the score
score = 0

# Font for displaying the score
score_font = pygame.font.Font(None, 36)

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 150)  # Update every 150 milliseconds

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == SCREEN_UPDATE:
            # Move the snake based on the timer event
            snake.move_snake()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if snake.direction != Vector2(0, 1):
                    snake.direction_queue.append(Vector2(0, -1))
            if event.key == pygame.K_DOWN:
                if snake.direction != Vector2(0, -1):
                    snake.direction_queue.append(Vector2(0, 1))
            if event.key == pygame.K_RIGHT:
                if snake.direction != Vector2(-1, 0):
                    snake.direction_queue.append(Vector2(1, 0))
            if event.key == pygame.K_LEFT:
                if snake.direction != Vector2(1, 0):
                    snake.direction_queue.append(Vector2(-1, 0))

            # Limit the direction queue size to avoid overfilling
            if len(snake.direction_queue) > 3:
                snake.direction_queue = snake.direction_queue[:3]

    draw_background()  # Draw the checkerboard background
    fruit.draw_fruit()
    snake.draw_snake()
    draw_score()  # Draw the score on the screen
    pygame.display.update()

    # Check for collision with fruit
    check_collision()

    # Check for failure (collision with walls or self)
    if check_fail():
        print("Game Over")
        pygame.quit()
        sys.exit()

    clock.tick(60)  # Cap the frame rate to 60 FPS for smooth rendering
