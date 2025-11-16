"""
Installation and Setup Script for Nokia Snake Game
Automatically installs all required dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def check_package(package):
    """Check if a package is already installed"""
    try:
        __import__(package.split('>=')[0].replace('-', '_'))
        return True
    except ImportError:
        return False

def main():
    """Main setup function"""
    print("🐍 Nokia Snake Game - Setup Script")
    print("=" * 50)
    
    # List of required packages
    packages = [
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0", 
        "numpy>=1.24.0",
        "pygame>=2.4.0"
    ]
    
    print("Checking and installing required packages...")
    print()
    
    all_success = True
    
    for package in packages:
        package_name = package.split('>=')[0]
        print(f"Checking {package_name}...")
        
        if not check_package(package_name):
            print(f"Installing {package}...")
            if not install_package(package):
                all_success = False
        else:
            print(f"✓ {package_name} already installed")
    
    print()
    print("=" * 50)
    
    if all_success:
        print("🎉 Setup completed successfully!")
        print()
        print("To start the game, run:")
        print("  python main.py")
        print()
        print("Game Controls:")
        print("- Move your hand up/down/left/right to control the snake")
        print("- Pinch thumb and index finger for speed boost")
        print("- Show 'UP' gesture when game over to restart")
        print("- Press ESC in game window to quit")
    else:
        print("❌ Some packages failed to install.")
        print("Please install them manually using:")
        print("  pip install -r requirements.txt")
    
    print()
    print("Make sure your webcam is connected and working!")

if __name__ == "__main__":
    main()