"""decision_server.py - socket-based EEG decision server."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import mne

import constants as const
from data_collection import DummyCollector, LSLCollector
from preprocessing import extract_features, create_inverse_operator
from classification import EEGClassifier


def sample_to_epoch_label(
    sample: np.ndarray,
    inlet_info: mne.Info,
    label: str,
    window: float,
) -> mne.Epochs:
    """Build one Epoch from raw data given the ground-truth *label*."""
    expected_len = int(inlet_info["sfreq"] * window)
    if sample.shape[1] < expected_len:
        raise ValueError(
            f"Insufficient samples for epoch: got {sample.shape[1]}, expected {expected_len}"
        )

    raw = mne.io.RawArray(sample, inlet_info, verbose=False)
    raw.filter(const.C.F_LOW, const.C.F_HIGH, verbose=False)
    raw.notch_filter(const.C.F_NOTCH, trans_bandwidth=2, verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)

    event_timestep = int(const.C.BEFORE_MARKER_TIME * inlet_info["sfreq"]) - 1
    if label == "right":
        event_id = {"MI_right": 1}
        events = np.array([[event_timestep, 0, 1]])
    else:
        event_id = {"MI_left": 0}
        events = np.array([[event_timestep, 0, 0]])

    tmin = -const.C.BEFORE_MARKER_TIME + 0.05
    tmax = const.C.MARKER_TIME

    epochs = mne.Epochs(
        raw,
        events,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        event_id=event_id,
        preload=True,
        verbose=False,
    )
    return epochs


class DecisionServer:
    """Socket server handling EEG collection and classification."""

    def __init__(self, host: str = "localhost", port: int = 8765, offline: bool = False) -> None:
        self.host = host
        self.port = port
        self.offline = offline
        # Use a 64-channel dummy collector so the pipeline matches the
        # pre-trained models based on the GIGA dataset.  Channel names also
        # mimic the real montage for compatibility with the inverse operator.
        self.collector = (
            DummyCollector(n_channels=len(const.C.CH_NAMES_64))
            if offline
            else LSLCollector()
        )
        self.label: str | None = None
        self.subject_number: int | None = None
        self.epochs: list[mne.Epochs] = []
        self.X: list[np.ndarray] = []
        self.Y: list[int] = []
        self.classifier: EEGClassifier | None = None
        self.inverse_operator: mne.minimum_norm.InverseOperator | None = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind((self.host, self.port))
        self._sock.listen(1)
        # Ensure accept() unblocks periodically so KeyboardInterrupt can be
        # raised even on platforms where signals don't interrupt system calls
        # (notably Windows).  This allows Ctrl+C to stop the server.
        self._sock.settimeout(1.0)
        print(f"[DecisionServer] Listening on {self.host}:{self.port}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_window(self, window: float) -> Tuple[np.ndarray, np.ndarray]:
        """Collect ``window`` seconds of data from the EEG stream."""
        if self.offline:
            # DummyCollector always returns a fixed-size window so ``window`` is
            # ignored here.  This keeps the server logic identical in offline
            # mode while avoiding attributes (``inlet``/``offset``) that only
            # exist on :class:`LSLCollector`.
            return self.collector.collect_window()

        sfreq = self.collector.info["sfreq"]
        n_samples = int(sfreq * window)
        sample_arr = np.empty((len(self.collector.info["ch_names"]), 0))
        ts_arr = np.empty(0)
        while sample_arr.shape[1] < n_samples:
            need = n_samples - sample_arr.shape[1]
            chunk, ts = self.collector.inlet.pull_chunk(timeout=window, max_samples=need)
            if not chunk:
                break
            sample_arr = np.hstack((sample_arr, np.asarray(chunk).T))
            ts_arr = np.concatenate((ts_arr, np.asarray(ts) + self.collector.offset))
        return sample_arr, ts_arr

    # ------------------------------------------------------------------
    def _save_data(self) -> None:
        if self.subject_number is None or not self.epochs:
            return
        folder = Path("data") / f"s{self.subject_number:02}"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H%M")
        mne.concatenate_epochs(self.epochs, verbose=False).save(
            folder / f"{stamp}_epo.fif"
        )
        if self.X:
            np.save(folder / f"{stamp}_features.npy", self.X)
        print(f"[DecisionServer] Saved data to {folder}")

    # ------------------------------------------------------------------
    def _process(self, conn: socket.socket) -> None:
        with conn:
            buff = b""
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                buff += data
                if b"\n" not in buff:
                    continue
                line, buff = buff.split(b"\n", 1)
                resp = self._handle_command(line.decode().strip())
                conn.sendall(resp.encode() + b"\n")

    def _handle_command(self, cmd: str) -> str:
        parts = cmd.split()
        if not parts:
            return "ERR"
        action = parts[0].upper()
        if action == "LOAD_SUBJECT" and len(parts) == 2:
            try:
                self.subject_number = int(parts[1])
                self.classifier = EEGClassifier.from_subject(self.subject_number)
                self.inverse_operator = self.classifier.inverse_operator
                return "OK"
            except FileNotFoundError:
                self.subject_number = int(parts[1])
                self.classifier = None
                self.inverse_operator = None
                return "NO_MODEL"
        if action == "SET_LABEL" and len(parts) == 2:
            if parts[1] not in {"left", "right"}:
                return "ERR"
            self.label = parts[1]
            return "OK"
        if action == "GET_PRED" and len(parts) == 2:
            if self.label is None:
                return "ERR"
            window = float(parts[1])
            sample, _ = self._collect_window(window)
            try:
                epoch = sample_to_epoch_label(sample, self.collector.info, self.label, window)
            except ValueError as err:
                return f"ERR {err}"
            self.epochs.append(epoch)
            self.Y.append(0 if self.label == "left" else 1)
            if self.inverse_operator is None:
                self.inverse_operator = create_inverse_operator(epoch.info)
            feat = extract_features(epoch, self.inverse_operator, -0.1, 1.4)
            self.X.append(feat)
            if self.classifier:
                prob_arr = self.classifier.predict_proba(feat.reshape(1, -1))[0]
                label = "right" if prob_arr[1] >= prob_arr[0] else "left"
                prob = max(prob_arr)
                return f"{label} {prob:.3f}"
            return "NO_MODEL 0.0"
        if action == "SAVE_TRAIN":
            self._save_data()
            return "OK"
        return "ERR"

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close underlying resources."""
        try:
            self._sock.close()
        except OSError:
            pass

    def serve_forever(self) -> None:
        try:
            while True:
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    continue
                threading.Thread(
                    target=self._process,
                    args=(conn,),
                    daemon=True,
                ).start()
        except KeyboardInterrupt:
            print("\n[DecisionServer] Shutting down...")
        finally:
            self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG Decision Server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--offline", action="store_true", help="Use DummyCollector")
    args = parser.parse_args()

    server = DecisionServer(args.host, args.port, args.offline)
    server.serve_forever()
