from __future__ import annotations

import random
import sys
import time
import threading
from typing import List, Tuple
import argparse
import keyboard
import pygame
1
import constants as const  # singleton is const.C
from decision_client import DecisionClient


#os.environ["SDL_VIDEO_WINDOW_POS"]="%d,%d" % (1920+960,-480)


# --------------------------------------------------------------------------- #
#                               Game class                                    #
# --------------------------------------------------------------------------- #
class DroneCatcherGame:
    """
    A simple “control the drone with your mind” game driven by real-time EEG.
    """

    # --------------------------------------------------------------------- #
    #  Construction                                                         #
    # --------------------------------------------------------------------- #
    def __init__(self) -> None:
        # --- subject id -------------------------------------------------- #
        self.subject_number: str = self._ask_subject_number()

        # --- pygame setup ------------------------------------------------ #
        pygame.init()
        self.screen = pygame.display.set_mode(
            (const.SCREEN_WIDTH, const.SCREEN_HEIGHT + const.LOAD_BAR_HEIGHT)
        )
        pygame.display.set_caption("Drone Catcher")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)

        # --- time variables --------------------------------------------- #
        self.before_marker_time = const.BEFORE_MARKER_TIME
        self.marker_time = const.MARKER_TIME
        self.total_time = const.TOTAL_TIME
        self.start_time = time.time()

        # --- game variables --------------------------------------------- #
        self.end_value = const.END_VALUE
        self.player_pos = [
            const.SCREEN_WIDTH // 2,
            const.SCREEN_HEIGHT - const.PLAYER_HEIGHT,
        ]

        # Position initiale du drone : centre
        self.drone_pos = [
            const.SCREEN_WIDTH // 2 - const.DRONE_SIZE // 2,
            const.SCREEN_HEIGHT // 2 - const.DRONE_SIZE // 2
        ]

        self.right_hand: str = "closed"
        self.left_hand: str = "closed"
        self.game_mode: str = "training"

        # --- images ------------------------------------------------------ #
        self._load_images()

        # --- LSL & EEG --------------------------------------------------- #
        self.client = DecisionClient()
        response = self.client.load_subject(int(self.subject_number))
        if response == "NO_MODEL":
            print("[DroneCatcher] No existing model, starting without classifier.")

        self.classifier_done = False
        self.classifier_started = False
        self.classifier_thread: threading.Thread | None = None
        self.classifier_result: Tuple[str, float] | None = None


        # Marker positions on the load bar
        self.first_marker = (
            const.SCREEN_WIDTH * self.before_marker_time / self.total_time
        )
        self.second_marker = (
            const.SCREEN_WIDTH
            * (self.before_marker_time + self.marker_time)
            / self.total_time
        )

        #déplacement du drone
        self.drone_target_x = self.drone_pos[0]  # Cible x
        self.drone_move_speed = 0                # Pixels par frame
        self.drone_moving = False                # Mouvement en cours


    # --------------------------------------------------------------------- #
    #  Helper methods                                                       #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _ask_subject_number() -> str:
        """Simple blocking prompt for an integer subject id."""
        while True:
            s = input("Enter the subject number: ")
            if s.isdigit():
                return s
            print("→ Must be an integer!")

    def _load_images(self) -> None:
        """Load and scale all png assets once."""

        self.tree_image = pygame.transform.scale(
            pygame.image.load(const.TREE_IMAGE_PATH).convert_alpha(),
            (
                int(pygame.image.load(const.TREE_IMAGE_PATH).get_width() * 3),
                int(pygame.image.load(const.TREE_IMAGE_PATH).get_height() * 2),
            ),
        )
        self.drone_image = self._load_and_scale(
            const.DRONE_IMAGE_PATH, size=const.DRONE_SIZE
        )

    @staticmethod
    def _load_and_scale(path: str, size: int | None = None):
        img = pygame.image.load(path).convert_alpha()
        if size:
            img = pygame.transform.scale(img, (size, size))
        else:
            img = pygame.transform.scale(
                img, (const.PLAYER_WIDTH, const.PLAYER_HEIGHT)
            )
        return img

    # --------------------------------------------------------------------- #
    #  Game logic                                                           #
    # --------------------------------------------------------------------- #

    def _draw(self) -> None:
        # Affiche l'image de fond 
        self.screen.blit(self.tree_image, (0, 0))

        # Affiche le drone
        self.screen.blit(self.drone_image, self.drone_pos)

        # Affiche éventuellement une ligne centrale pour repère
        center_x = const.SCREEN_WIDTH // 2
        pygame.draw.line(self.screen, (200, 200, 200), (center_x, 0), (center_x, const.SCREEN_HEIGHT), 1)

        pygame.display.flip()



    def _classify(self) -> Tuple[str, float]:
        if self.game_mode == "offline":
            prob = random.random()
            label = "right" if prob > 0.5 else "left"
            print(f"[DroneCatcher] Received decision {label} (prob={prob:.2f})")
            return label, 1

        label, prob = self.client.get_pred(const.SAMPLE_WINDOW)
        print(f"[DroneCatcher] Received decision {label} (prob={prob:.2f})")
        if self.game_mode == "training":
            # Ignore prediction during training
            label = "right" if self.drone_pos[0] > const.SCREEN_WIDTH / 2 else "left"
            prob = 1
            return label, prob

        if self.game_mode == "test":
            # In test mode, we just return the prediction
            return label, prob



    def _classify_async(self) -> None:
        """Run classification in a background thread."""
        self.classifier_result = self._classify()


    def _move_drone(self, direction: str) -> None:
        step_distance = 200  # distance totale à parcourir (en pixels)
        duration = 2.0        # durée du déplacement en secondes
        fps = const.FPS
        frames = int(duration * fps)

        if direction == "left":
            self.drone_target_x = max(0, self.drone_pos[0] - step_distance)
        elif direction == "right":
            self.drone_target_x = min(
                const.SCREEN_WIDTH - const.DRONE_SIZE,
                self.drone_pos[0] + step_distance,
            )

        self.drone_move_speed = (self.drone_target_x - self.drone_pos[0]) / frames
        self.drone_moving = True

    def perform_takeoff(self) -> None:
        """Anime le décollage du drone pendant 3 secondes."""
        start_y = const.SCREEN_HEIGHT  # départ en bas de l'écran
        target_y = const.SCREEN_HEIGHT // 2 - const.DRONE_SIZE // 2  # milieu écran verticalement
        duration = 3.0  # durée en secondes

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > duration:
                break

            progress = elapsed / duration
            current_y = start_y - (start_y - target_y) * progress

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Affiche le fond
            self.screen.blit(self.tree_image, (0, 0))

            # Affiche le drone en montée
            self.screen.blit(self.drone_image, (self.drone_pos[0], int(current_y)))

            pygame.display.flip()
            self.clock.tick(const.FPS)

        # Mise à jour finale de la position verticale du drone
        self.drone_pos[1] = target_y

    def perform_landing(self) -> None:
        """Anime l’atterrissage de la pomme quand Espace est pressé."""
        start_y = self.drone_pos[1]  # Position actuelle (milieu)
        target_y = const.SCREEN_HEIGHT  # Bas de l’écran (hors champ)

        duration = 3.0  # durée en secondes
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > duration:
                break

            progress = elapsed / duration
            current_y = start_y + (target_y - start_y) * progress

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            # Affiche le fond
            self.screen.blit(self.tree_image, (0, 0))

            # Affiche la pomme en descente
            self.screen.blit(self.drone_image, (self.drone_pos[0], int(current_y)))

            pygame.display.flip()
            self.clock.tick(const.FPS)

        # Mise à jour finale de la position verticale de la pomme
        self.drone_pos[1] = target_y

    def _update(self):
        if self.drone_moving:
            # Avancer le drone petit à petit
            self.drone_pos[0] += self.drone_move_speed

            # Arrêter si la cible est atteinte (ou dépassée)
            if (self.drone_move_speed > 0 and self.drone_pos[0] >= self.drone_target_x) or \
               (self.drone_move_speed < 0 and self.drone_pos[0] <= self.drone_target_x):
                self.drone_pos[0] = self.drone_target_x
                self.drone_moving = False


    # --------------------------------------------------------------------- #
    #  Data persistence                                                     #
    # --------------------------------------------------------------------- #
    def _save_data(self) -> None:
        self.client.save_train()

    # --------------------------------------------------------------------- #
    #  Main loop                                                            #
    # --------------------------------------------------------------------- #
    def run(self) -> None:
        self.perform_takeoff()
        self.start_time = time.time()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.perform_landing()
                        pygame.quit()
                        sys.exit()

            if time.time() - self.start_time > 1.5:
                label, prob = self._classify()
                if prob >= const.CONFIDENCE_THRESHOLD:
                    self._move_drone(label)
                self.start_time = time.time()
            self._update()
            self._draw()
            self.clock.tick(const.FPS)

    # --------------------------------------------------------------------- #
    #  (Optional) simple start menu – unchanged for brevity                 #
    # --------------------------------------------------------------------- #
    # You can keep your existing show_menu() here if you want to preserve
    # the graphical selection logic. It only needs constant renaming
    # (SCREEN_WIDTH → const.C.SCREEN_WIDTH, etc.). Omitted for clarity.


# --------------------------------------------------------------------------- #
#                                 Entrypoint                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone-Catcher EEG game")
    parser.add_argument(
        "--mode",
        choices=["training", "test", "offline"],
        default="training",
        help="Select gameplay mode (default: training).",
    )
    args = parser.parse_args()

    game = DroneCatcherGame()
    game.game_mode = args.mode

    print(f"\nGame started in **{game.game_mode.upper()}** mode!\n")
    game.run()

