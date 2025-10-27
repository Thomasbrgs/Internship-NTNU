==============================
Drone Game IRL - Setup Guide
==============================

This project lets you control a Crazyflie drone using EEG-based decisions, either simulated (offline) or real-time.
It includes a decision server (`decision_server.py`) and a game controller (`main_menu.py`).


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
        libffi-dev


2. Clone the Git Repository
-------------------------------

    git clone https://github.com/<your-username>/<your-project>.git
    cd <your-project>

(Replace `<your-username>` and `<your-project>` with the actual values.)


3. Set Up Python Virtual Environment
----------------------------------------

    python3 -m venv mon_env
    source mon_env/bin/activate

Then install the required Python packages:

    pip install -r requirements.txt


4. Crazyradio USB Dongle (optional but required for real drone)
------------------------------------------------------------------

Create a udev rule to allow USB access:

    sudo nano /etc/udev/rules.d/99-crazyradio.rules

Paste the following content:

    SUBSYSTEM=="usb", ATTR{idVendor}=="1915", ATTR{idProduct}=="7777", MODE="0666"

Reload the rules:

    sudo udevadm control --reload-rules
    sudo udevadm trigger


5. Launch EEG Decision Server (offline mode)
------------------------------------------------

In one terminal:

    source mon_env/bin/activate
    python decision_server.py --offline


6. Launch the Drone Game
----------------------------

In a second terminal:

    source mon_env/bin/activate
    python drone_game_irl.py

You will be asked to select a game mode:
- `training` – collect data for classifier
- `test` – use the model for predictions
- `free` – free control with EEG or keyboard

Keyboard Controls:
    Z/Q/S/D  : move
    Q/E      : rotate
    R/F      : up/down
    Space    : stop
    Esc      : exit



