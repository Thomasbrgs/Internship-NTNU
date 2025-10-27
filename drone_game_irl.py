from __future__ import annotations
import sys
import random
import logging
import time
import threading
import queue

from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

import keyboard
import argparse

import constants as const
from decision_client import DecisionClient

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E703')
DEFAULT_HEIGHT = 0.5
deck_attached_event = Event()
logging.basicConfig(level=logging.ERROR)

# Commandes possibles
COMMAND_FORWARD = 'forward'
COMMAND_BACKWARD = 'backward'
COMMAND_LEFT = 'left'
COMMAND_RIGHT = 'right'
COMMAND_TURN_LEFT = 'turn_left'
COMMAND_TURN_RIGHT = 'turn_right'
COMMAND_UP = 'up'
COMMAND_DOWN = 'down'
COMMAND_STOP = 'stop'
COMMAND_EXIT = 'exit'

# Mappage clavier
KEY_COMMAND_MAP = {
    'z': COMMAND_FORWARD,
    's': COMMAND_BACKWARD,
    'q': COMMAND_LEFT,
    'd': COMMAND_RIGHT,
    'q': COMMAND_TURN_LEFT,
    'e': COMMAND_TURN_RIGHT,
    'r': COMMAND_UP,
    'f': COMMAND_DOWN,
    'x': COMMAND_STOP,
    'space': COMMAND_EXIT
}
class DroneCatcherControl:
    """
    Drone control logic using EEG-based predictions, without graphical interface.
    """

    def __init__(self) -> None:
        # --- Subject & mode --- #
        self.subject_number: str = self._ask_subject_number()
        self.game_mode: str = "training"

        # --- EEG Classifier --- #
        self.client = DecisionClient()
        response = self.client.load_subject(int(self.subject_number))
        if response == "NO_MODEL":
            print("[DroneCatcher] No existing model, starting without classifier.")

        self.classifier_result: tuple[str, float] | None = None
        self.start_time = time.time()

        # --- Drone state --- #
        self.drone_pos_x = const.SCREEN_WIDTH // 2

    @staticmethod
    def _ask_subject_number() -> str:
        """Simple blocking prompt for an integer subject id."""
        while True:
            s = input("Enter the subject number: ")
            if s.isdigit():
                return s
            print("→ Must be an integer!")

    def _classify(self) -> tuple[str, float]:
        if self.game_mode == "offline":
            prob = random.random()
            label = "right" if prob > 0.5 else "left"
            print(f"[DroneCatcher] Simulated decision: {label} (prob={prob:.2f})")
            return label, 1.0

        label, prob = self.client.get_pred(const.SAMPLE_WINDOW)
        print(f"[DroneCatcher] EEG decision: {label} (prob={prob:.2f})")

        if self.game_mode == "training":
            label = "right" if self.drone_pos_x > const.SCREEN_WIDTH / 2 else "left"
            prob = 1.0

        return label, prob

    def _save_data(self) -> None:
        self.client.save_train()
        print("[DroneCatcher] Training data saved.")

    def push_decision_to_queue(self, command_queue: queue.Queue) -> None:
        label, prob = self._classify()

        if prob < const.CONFIDENCE_THRESHOLD:
            print("[DroneCatcher] Prediction confidence too low, ignoring.")
            return

        if label == "left":
            command_queue.put(COMMAND_LEFT)
        elif label == "right":
            command_queue.put(COMMAND_RIGHT)
        else:
            print(f"[DroneCatcher] Unknown label: {label}")

    def run(self) -> None:
        print(f"[DroneCatcher] Running in {self.game_mode.upper()} mode.\n")
        try:
            while True:
                time.sleep(1.5)

                label, prob = self._classify()
                if prob >= const.CONFIDENCE_THRESHOLD:
                    self._move_drone(label)

        except KeyboardInterrupt:
            print("\n[DroneCatcher] Interrupted. Saving data and exiting.")
            if self.game_mode == "training":
                self._save_data()
            sys.exit()

def param_deck_flow(name, value_str):
    value = int(value_str)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')

def keyboard_listener(command_queue, exit_event):
    print("Contrôle clavier actif. Z/Q/S/D pour se déplacer, Q/E tourner, R/F monter/descendre, Space arrêt, Esc pour quitter.")
    try:
        while not exit_event.is_set():
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                key = event.name
                if key in KEY_COMMAND_MAP:
                    command = KEY_COMMAND_MAP[key]
                    command_queue.put(command)
                    if command == COMMAND_EXIT:
                        exit_event.set()
                        break
    except:
        pass

def move_with_commands(scf, command_queue, exit_event, test_mode):
    if test_mode:
        while not exit_event.is_set():
            try:
                command = command_queue.get(timeout=0.1)
                print(f"Commande simulée : {command}")
                if command == COMMAND_EXIT:
                    exit_event.set()
            except queue.Empty:
                continue
    else:
        with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
            print("Drone connecté. En attente de commandes...")

            while not exit_event.is_set():
                try:
                    command = command_queue.get(timeout=0.1)
                    print(f"Commande reçue : {command}")

                    if command == COMMAND_FORWARD:
                        mc.forward(0.2)
                    elif command == COMMAND_BACKWARD:
                        mc.back(0.2)
                    elif command == COMMAND_LEFT:
                        mc.left(0.2)
                    elif command == COMMAND_RIGHT:
                        mc.right(0.2)
                    elif command == COMMAND_TURN_LEFT:
                        mc.turn_left(15)
                    elif command == COMMAND_TURN_RIGHT:
                        mc.turn_right(15)
                    elif command == COMMAND_UP:
                        mc.up(0.2)
                    elif command == COMMAND_DOWN:
                        mc.down(0.2)
                    elif command == COMMAND_STOP:
                        mc.stop()
                    elif command == COMMAND_EXIT:
                        exit_event.set()
                    else:
                        print(f"Commande inconnue : {command}")
                except queue.Empty:
                    continue

            mc.stop()
            print("MotionCommander arrêté.")




def eeg_decision_loop(controller: DroneCatcherControl, command_queue: queue.Queue, exit_event: threading.Event):
    while not exit_event.is_set():
        controller.push_decision_to_queue(command_queue)
        time.sleep(1.5)  # même cadence que dans le jeu original


def main():
    cflib.crtp.init_drivers()
    test_mode = False
    command_queue = queue.Queue()
    exit_event = threading.Event()

    # Démarrage du thread clavier
    keyboard_thread = threading.Thread(target=keyboard_listener, args=(command_queue, exit_event), daemon=True)
    keyboard_thread.start()

        # Démarrer EEG → commande vers queue
    controller = DroneCatcherControl()
    controller.game_mode = "training"  # ou "test", selon l’usage

    eeg_thread = threading.Thread(
        target=eeg_decision_loop,
        args=(controller, command_queue, exit_event),
        daemon=True
    )
    eeg_thread.start()


    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            scf.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=param_deck_flow)
            time.sleep(1)

            if not deck_attached_event.wait(timeout=5):
                print('Deck non détecté. Passage en mode test.')
                test_mode = True

            move_with_commands(scf, command_queue, exit_event, test_mode)

    except Exception as e:
        print(f"Connexion au drone échouée : {e}. Mode test activé.")
        test_mode = True
        move_with_commands(None, command_queue, exit_event, test_mode)

    finally:
        print("Fermeture propre...")
        try:
            keyboard_thread.join()
        except:
            pass
        print("Programme terminé.")

if __name__ == '__main__':
    main()
