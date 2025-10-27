==============================
DroneCatcher EEG Game - Setup Guide
==============================

This project lets you control a virtual drone (with Pygame) using EEG-based decisions,
either in offline (simulated) mode or real-time (online) via the decision_server.py.

It includes:
- A decision server (decision_server.py) for EEG signal processing & classification
- A game client (drone_catcher_game.py) for interactive drone control


1. Install System Dependencies (Ubuntu/Debian)
--------------------------------------------------

Open a terminal and run:

    sudo apt update && sudo apt install -y \
        libusb-1.0-0-dev \
        python3 \
        python3-pip \
        python3-venv \
        libglib2.0-dev \
        build-essential \
        libffi-dev \
        libsdl2-dev \
        libsdl2-image-dev


2. Clone the Git Repository
-------------------------------

    git clone https://github.com/<your-username>/<your-project>.git
    cd <your-project>

(Replace <your-username> and <your-project> with the actual values.)


3. Set Up Python Virtual Environment
----------------------------------------

    python3 -m venv venv
    source venv/bin/activate

Then install the required Python packages:

    pip install -r requirements.txt



5. Launch EEG Decision Server
---------------------------------

In one terminal:

    source venv/bin/activate
    python decision_server.py --offline

Options:
    --offline            simulate predictions (no EEG stream required)
    --dataset giga|stroke   replay a dataset for testing classifiers
    --classifier rf|lda|spec_cnn   choose classifier type


6. Launch the DroneCatcher Game
-----------------------------------

In another terminal:

    source venv/bin/activate
    python drone_catcher_game.py --mode test

Modes:
    training   collect training data (randomized left/right labels)
    test       use the trained model for predictions
    offline    simulate decisions randomly (no EEG required)


7. Controls
----------------
Game:
    - Drone moves automatically based on EEG (or offline/random mode)
    - Press Space   trigger landing & exit
    - Press Esc     quit immediately

Classifier Training:
    - Labels alternate between left and right
    - Model updates after training


You’re ready to start flying your EEG-controlled drone (virtual or real)!
