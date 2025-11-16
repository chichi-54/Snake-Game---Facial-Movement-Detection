"""
Main entry point for Nokia Snake Game with Gesture Control (updated for YOLO-based GestureController)

Notes:
- Set environment variables HAND_MODEL and FACE_MODEL if you want to point to specific checkpoints.
  Example:
    export HAND_MODEL=/path/to/yolov8n-handpose.pt
    export FACE_MODEL=/path/to/yolov8n-face-lindevs.pt

- Install required packages:
    pip install ultralytics opencv-python numpy pygame

- If models are missing or fail to load, game still runs without gesture input (fallback).
"""

import os
import cv2
import pygame
import threading
import time
import traceback

from gesture_controller import GestureController
from snake_game import SnakeGame


class GameManager:
    def __init__(self):
        """Initialize the game manager"""
        pygame.init()
        pygame.display.set_caption("Nokia Snake - Game")
        self.game = SnakeGame()

        # Read model paths from environment variables (or use defaults)
        hand_model = os.environ.get("HAND_MODEL", "yolov8n-pose.pt")
        face_model = os.environ.get("FACE_MODEL", "yolov8n-face-lindevs.pt")  # set to None if you don't want face boxes
        device = os.environ.get("GC_DEVICE", "cpu")  # "cpu" or "cuda"

        # Try to create the gesture controller; if it fails, we continue with no-gesture mode
        self.gesture_controller = None
        try:
            print(f"Loading GestureController (hand_model={hand_model}, face_model={face_model}, device={device})")
            self.gesture_controller = GestureController(hand_model=hand_model, face_model=face_model, device=device)
            print("GestureController loaded successfully.")
        except Exception as e:
            print("Warning: Failed to initialize GestureController. Running without gestures.")
            traceback.print_exc()
            self.gesture_controller = None

        self.cap = None
        self.running = True
        self.gesture_thread = None

        # Gesture state
        self.current_gesture = None
        self.is_speed_boost = False

    def initialize_camera(self):
        """Initialize the webcam"""
        # If no gesture controller, we don't need to open a camera
        if not self.gesture_controller:
            return True

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True

    def gesture_detection_loop(self):
        """Main loop for gesture detection (runs in separate thread)"""
        # If no gesture controller, just exit this thread immediately
        if not self.gesture_controller or self.cap is None:
            return

        # Create a named window for the gesture feed so the user can close it
        cv2.namedWindow('Nokia Snake - Gesture Control', cv2.WINDOW_NORMAL)
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                # small sleep to avoid busy loop on camera failure
                time.sleep(0.05)
                continue

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect gestures
            try:
                gesture, pinch, annotated_frame = self.gesture_controller.detect_gestures(frame)
            except Exception as e:
                # If the model throws during runtime, log and stop gesture processing
                print("Gesture detection error, disabling gesture controller:", e)
                traceback.print_exc()
                self.gesture_controller = None
                break

            # Update gesture state
            self.current_gesture = gesture
            self.is_speed_boost = bool(pinch)

            # Display gesture window
            try:
                cv2.imshow('Nokia Snake - Gesture Control', annotated_frame)
            except Exception:
                # ignore imshow errors (window closed externally)
                pass

            # Handle window close or 'q' pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                # Signal the whole app to stop
                self.running = False
                break

            time.sleep(0.033)  # ~30 FPS

        # If we exit loop, ensure camera released (cleanup() will also handle it)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        cv2.destroyWindow('Nokia Snake - Gesture Control')

    def run(self):
        """Main game loop"""
        if not self.initialize_camera():
            print("Failed to initialize camera. Continuing without gestures...")

        # Start gesture detection in separate thread (only if we have a camera + controller)
        if self.gesture_controller and self.cap is not None:
            self.gesture_thread = threading.Thread(target=self.gesture_detection_loop, name="GestureThread")
            self.gesture_thread.daemon = True
            self.gesture_thread.start()

        print("Nokia Snake Game Started!")
        print("Controls:")
        print("- Move your hand up/down/left/right to control the snake (if gestures available)")
        print("- Pinch thumb and index finger for speed boost (if gestures available)")
        print("- Show 'UP' gesture when game over to restart")
        print("- Press ESC in game window to quit")
        print("\nGame Window: Classic Nokia Snake")
        print("Gesture Window: Webcam feed with hand tracking (press 'q' to close)")

        # Main game loop
        last_update = time.time()

        # Ensure game created any necessary pygame resources
       # self.game.reset_if_needed()

        try:
            while self.running:
                current_time = time.time()

                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False

                # Apply gesture controls (if available)
                if self.current_gesture:
                    # change_direction should ignore invalid or reverse moves internally
                    self.game.change_direction(self.current_gesture)
                    if self.game.game_over:
                        # if game over, allow restart by showing UP gesture
                        self.game.handle_restart(self.current_gesture)

                # Apply speed boost
                self.game.set_speed_boost(self.is_speed_boost)

                # Update game at appropriate speed
                target_fps = self.game.get_current_speed()
                if current_time - last_update >= 1.0 / max(1, target_fps):
                    self.game.update()
                    last_update = current_time

                # Draw game
                self.game.draw()
                # Cap the display framerate for smoothness
                self.game.clock.tick(60)  # Display refresh rate

        except KeyboardInterrupt:
            print("\nGame interrupted by user")
        except Exception as e:
            print(f"An error occurred in game loop: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        # Stop loops
        self.running = False

        # Release camera if not already released
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # Close all OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # Let gesture thread finish
        if self.gesture_thread and self.gesture_thread.is_alive():
            self.gesture_thread.join(timeout=1.0)

        # Quit the game
        try:
            self.game.quit()
        except Exception:
            pass

        # If gesture controller has resources to free, you can add them here (e.g., model.cleanup())
        print("Game closed successfully!")

def main():
    """Main function"""
    try:
        game_manager = GameManager()
        game_manager.run()
    except Exception as e:
        print(f"Fatal error starting the game: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
