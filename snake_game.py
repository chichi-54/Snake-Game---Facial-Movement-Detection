"""
Nokia Snake Game Implementation
Classic snake game with Nokia-style graphics and mechanics
"""

import pygame
import random
import math
from typing import List, Tuple, Optional
from enum import Enum

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class SnakeGame:
    def __init__(self, width: int = 600, height: int = 600):
        """Initialize the Nokia Snake game"""
        # Game settings
        self.width = width
        self.height = height
        self.grid_size = 20
        self.grid_width = width // self.grid_size
        self.grid_height = height // self.grid_size
        
        # Colors (Nokia green theme)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.NOKIA_GREEN = (155, 188, 15)
        self.DARK_GREEN = (139, 172, 15)
        self.LIGHT_GREEN = (204, 255, 51)
        self.RED = (255, 0, 0)
        self.ORANGE = (255, 165, 0)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Nokia Snake - Gesture Control")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Game state
        self.reset_game()
        
        # Particle effects
        self.particles = []
        
    def reset_game(self):
        """Reset game to initial state"""
        # Snake initialization
        start_x = self.grid_width // 2
        start_y = self.grid_height // 2
        self.snake = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        
        # Game state
        self.score = 0
        self.game_over = False
        self.speed_boost = False
        self.base_speed = 8
        self.boost_speed = 15
        
        # Spawn first fruit
        self.spawn_fruit()
        
    def spawn_fruit(self):
        """Spawn a new fruit at random location"""
        while True:
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            if (x, y) not in self.snake:
                self.fruit = (x, y)
                break
    
    def add_particle_effect(self, x: int, y: int):
        """Add particle effect when eating fruit"""
        for _ in range(8):
            particle = {
                'x': x * self.grid_size + self.grid_size // 2,
                'y': y * self.grid_size + self.grid_size // 2,
                'vx': random.uniform(-3, 3),
                'vy': random.uniform(-3, 3),
                'life': 30,
                'max_life': 30
            }
            self.particles.append(particle)
    
    def update_particles(self):
        """Update particle effects"""
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
    
    def change_direction(self, new_direction: str):
        """Change snake direction based on gesture input"""
        if self.game_over:
            return
            
        direction_map = {
            "UP": Direction.UP,
            "DOWN": Direction.DOWN,
            "LEFT": Direction.LEFT,
            "RIGHT": Direction.RIGHT
        }
        
        if new_direction in direction_map:
            new_dir = direction_map[new_direction]
            # Prevent immediate reversal
            opposite = {
                Direction.UP: Direction.DOWN,
                Direction.DOWN: Direction.UP,
                Direction.LEFT: Direction.RIGHT,
                Direction.RIGHT: Direction.LEFT
            }
            
            if new_dir != opposite.get(self.direction):
                self.next_direction = new_dir
    
    def set_speed_boost(self, boost: bool):
        """Set speed boost state"""
        self.speed_boost = boost
    
    def update(self):
        """Update game state"""
        if self.game_over:
            return
        
        # Update direction
        self.direction = self.next_direction
        
        # Move snake
        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or 
            new_head[1] < 0 or new_head[1] >= self.grid_height):
            self.game_over = True
            return
        
        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            return
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check fruit collision
        if new_head == self.fruit:
            self.score += 10
            self.add_particle_effect(new_head[0], new_head[1])
            self.spawn_fruit()
        else:
            # Remove tail if no fruit eaten
            self.snake.pop()
        
        # Update particles
        self.update_particles()
    
    def draw_snake_segment(self, surface, x: int, y: int, is_head: bool = False):
        """Draw a single snake segment with Nokia-style appearance"""
        pixel_x = x * self.grid_size
        pixel_y = y * self.grid_size
        
        # Main segment rectangle
        segment_rect = pygame.Rect(pixel_x + 1, pixel_y + 1, 
                                 self.grid_size - 2, self.grid_size - 2)
        
        if is_head:
            # Head is slightly different color
            pygame.draw.rect(surface, self.LIGHT_GREEN, segment_rect)
            # Draw eyes
            eye_size = 3
            left_eye = pygame.Rect(pixel_x + 5, pixel_y + 5, eye_size, eye_size)
            right_eye = pygame.Rect(pixel_x + 12, pixel_y + 5, eye_size, eye_size)
            pygame.draw.rect(surface, self.BLACK, left_eye)
            pygame.draw.rect(surface, self.BLACK, right_eye)
        else:
            # Body segments
            pygame.draw.rect(surface, self.NOKIA_GREEN, segment_rect)
        
        # Border for 3D effect
        pygame.draw.rect(surface, self.DARK_GREEN, segment_rect, 1)
    
    def draw_fruit(self, surface):
        """Draw fruit with glowing effect"""
        x, y = self.fruit
        pixel_x = x * self.grid_size
        pixel_y = y * self.grid_size
        
        # Glowing effect
        glow_size = self.grid_size + 4
        glow_rect = pygame.Rect(pixel_x - 2, pixel_y - 2, glow_size, glow_size)
        pygame.draw.rect(surface, self.ORANGE, glow_rect, 2)
        
        # Main fruit
        fruit_rect = pygame.Rect(pixel_x + 2, pixel_y + 2, 
                               self.grid_size - 4, self.grid_size - 4)
        pygame.draw.rect(surface, self.RED, fruit_rect)
        
        # Shine effect
        shine_rect = pygame.Rect(pixel_x + 4, pixel_y + 4, 4, 4)
        pygame.draw.rect(surface, self.WHITE, shine_rect)
    
    def draw_particles(self, surface):
        """Draw particle effects"""
        for particle in self.particles:
            alpha = int(255 * (particle['life'] / particle['max_life']))
            size = max(1, int(3 * (particle['life'] / particle['max_life'])))
            
            # Create a surface for the particle with alpha
            particle_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            color = (*self.ORANGE, alpha)
            pygame.draw.circle(particle_surface, color, (size, size), size)
            
            surface.blit(particle_surface, 
                        (int(particle['x'] - size), int(particle['y'] - size)))
    
    def draw_grid(self, surface):
        """Draw Nokia-style grid background"""
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(surface, (40, 40, 40), (x, 0), (x, self.height))
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(surface, (40, 40, 40), (0, y), (self.width, y))
    
    def draw(self):
        """Draw the game"""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw grid
        self.draw_grid(self.screen)
        
        if not self.game_over:
            # Draw snake
            for i, segment in enumerate(self.snake):
                self.draw_snake_segment(self.screen, segment[0], segment[1], i == 0)
            
            # Draw fruit
            self.draw_fruit(self.screen)
            
            # Draw particles
            self.draw_particles(self.screen)
        
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
    
    def draw_ui(self):
        """Draw user interface elements"""
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Speed boost indicator
        if self.speed_boost:
            boost_text = self.small_font.render("SPEED BOOST!", True, self.LIGHT_GREEN)
            self.screen.blit(boost_text, (10, 50))
        
        # Game over screen
        if self.game_over:
            # Semi-transparent overlay
            overlay = pygame.Surface((self.width, self.height))
            overlay.set_alpha(128)
            overlay.fill(self.BLACK)
            self.screen.blit(overlay, (0, 0))
            
            # Game over text
            game_over_text = self.font.render("GAME OVER", True, self.WHITE)
            final_score_text = self.font.render(f"Final Score: {self.score}", True, self.WHITE)
            restart_text = self.small_font.render("Show 'UP' gesture to restart", True, self.NOKIA_GREEN)
            
            # Center the text
            go_rect = game_over_text.get_rect(center=(self.width//2, self.height//2 - 40))
            fs_rect = final_score_text.get_rect(center=(self.width//2, self.height//2))
            rs_rect = restart_text.get_rect(center=(self.width//2, self.height//2 + 40))
            
            self.screen.blit(game_over_text, go_rect)
            self.screen.blit(final_score_text, fs_rect)
            self.screen.blit(restart_text, rs_rect)
    
    def handle_restart(self, gesture: Optional[str]):
        """Handle game restart"""
        if self.game_over and gesture == "UP":
            self.reset_game()
    
    def get_current_speed(self) -> int:
        """Get current game speed"""
        return self.boost_speed if self.speed_boost else self.base_speed
    
    def quit(self):
        """Quit the game"""
        pygame.quit()