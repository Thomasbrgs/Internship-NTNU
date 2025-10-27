"""Simple socket client to communicate with decision_server."""
from __future__ import annotations

import socket
from typing import Tuple


class DecisionClient:
    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        self.sock = socket.create_connection((host, port))
        self.file = self.sock.makefile("rwb")

    def _send(self, message: str) -> str:
        self.file.write(message.encode() + b"\n")
        self.file.flush()
        reply = self.file.readline().decode().strip()
        return reply

    def load_subject(self, subject: int) -> str:
        return self._send(f"LOAD_SUBJECT {subject}")

    def set_label(self, label: str) -> None:
        self._send(f"SET_LABEL {label}")

    def get_pred(self, window: float) -> Tuple[str, float]:
        resp = self._send(f"GET_PRED {window}")
        parts = resp.split()
        if len(parts) == 2:
            return parts[0], float(parts[1])
        return "right", 0.0

    def save_train(self) -> None:
        self._send("SAVE_TRAIN")

    def close(self) -> None:
        self.file.close()
        self.sock.close()