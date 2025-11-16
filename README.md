
<img width="2640" height="1485" alt="Image" src="https://github.com/user-attachments/assets/f46dcd35-8d04-4935-a2f8-73e8b141fbb3" />

---
# 🐍 Nokia Snake Game – YOLO Face and Hand Gesture Control

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.6+-green.svg)](https://www.pygame.org/)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-orange.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



---

## ✨ Overview

This project is a **modern twist on the classic Nokia Snake game**, inspired by Tuba Khan’s MediaPipe-based gesture version.

* I’ve replaced MediaPipe with **Ultralytics YOLOv8** models for **hand pose and face detection**.
* Gesture detection allows you to **control the snake with your hand** — swipe to move, pinch for a speed boost.
* Classic Nokia Snake gameplay is preserved, with a **grid-based snake**, fruit collection, collision detection, and score tracking.

This is **my own version**, built on inspiration but with **custom improvements** and YOLO integration. 🐍💨

---

## 🎮 Features

### **Classic Snake Gameplay**

* Authentic **Nokia-style green monochrome graphics**
* Grid-based snake movement
* Snake grows when eating fruit
* Collision detection with walls and self
* Score tracking and game-over screen

### **Gesture Controls**

* Hand movements detected via **YOLOv8 pose models**
* Pinch gesture (thumb + index) triggers **speed boost**
* Real-time webcam feed shows landmarks and current direction
* Optional face detection for visual feedback

### **Dual Window Interface**

* **Game Window**: Classic snake gameplay
* **Gesture Window**: Live webcam feed with hand tracking

---

## 🚀 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/nokia-snake-gesture.git
cd nokia-snake-gesture
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download YOLO Models

* Hand pose: [`yolov8n-pose.pt`](https://ultralytics.com/)
* Face detection: [`yolov8n-face.onnx`](https://ultralytics.com/)

> Place these files in your project root or update paths in `gesture_controller.py`.

---

## 🎮 Usage

```bash
python main.py
```

### Controls

| Gesture                         | Action            |
| ------------------------------- | ----------------- |
| Swipe Up                        | Move snake up     |
| Swipe Down                      | Move snake down   |
| Swipe Left                      | Move snake left   |
| Swipe Right                     | Move snake right  |
| Pinch (thumb + index)           | Speed boost       |
| Show UP gesture after game over | Restart game      |
| ESC                             | Quit game         |
| Q in gesture window             | Close webcam feed |

---

## 🛠 Technical Details

* **Game Engine:** Pygame, grid-based movement, 60 FPS refresh
* **Gesture Detection:** YOLOv8 pose for hands, YOLOv8 face for visualization
* **Threaded Architecture:** Gesture detection runs in parallel with the game
* **Optimizations:** Speed boost, cooldowns, and swipe detection thresholds

---

## 🌟 Acknowledgements

* **Inspired by Tuba Khan’s MediaPipe Nokia Snake project** – huge thanks for the inspiration! 🙏
* **Ultralytics YOLOv8** for pose and face detection
* **Pygame** for classic game rendering
* Classic **Nokia Snake** for nostalgia

---

## ⚙️ Customization

* **Gesture sensitivity:** Adjust `gesture_threshold` in `gesture_controller.py`
* **Game speed:** Modify `base_speed` and `boost_speed` in `snake_game.py`
* **Snake colors:** Change `NOKIA_GREEN` and `LIGHT_GREEN` constants in `snake_game.py`

---

## 📄 License

* **This project:** MIT License
* **YOLO models:** AGPL-3.0, download separately to comply with their license

---

## 💡 Fun Fact / Sprinkle

Playing with gestures feels like controlling the snake **with magic!** 🪄

* Swipe your hand, pinch to dash, and try to beat your high score.
* Classic Nokia vibes + modern AI = endless fun.

