"""
classification.py  –  Train, evaluate and persist the EEG classifier.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

import mne
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import create_inverse_operator


# --------------------------------------------------------------------------- #
#                              Helper free function                            #
# --------------------------------------------------------------------------- #
def check_for_existing_training_data(subject_number: int) -> bool:
    """
    Return True if at least one pair of .fif / .npy files exists for the subject.
    """
    folder = Path(f"data/s{subject_number:02}")
    return bool(list(folder.glob("*.fif"))) and bool(list(folder.glob("*.npy")))


# --------------------------------------------------------------------------- #
#                                 Main class                                  #
# --------------------------------------------------------------------------- #
class EEGClassifier:
    """
    Wrapper around a scikit-learn Pipeline (StandardScaler → PCA → LDA)
    plus the MNE inverse operator needed for feature extraction.
    """

    def __init__(
        self,
        pipeline,
        inverse_operator: mne.minimum_norm.InverseOperator | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.inverse_operator = inverse_operator

    # --------------------------------------------------------------------- #
    #  Class helpers                                                         #
    # --------------------------------------------------------------------- #
    @classmethod
    def from_subject(cls, subject_number: int) -> "EEGClassifier":
        """
        Load all training data for *subject_number*, fit the pipeline,
        and build the inverse operator.
        """
        folder = Path(f"data/s{subject_number:02}")
        fif_files = sorted(folder.glob("*.fif"))
        npy_files = sorted(folder.glob("*.npy"))

        if not fif_files or not npy_files:
            raise FileNotFoundError(
                f"No training data found for subject {subject_number:02}. "
                "Run the game in training mode first."
            )
        if len(fif_files) != len(npy_files):
            raise RuntimeError("Mismatch in the number of .fif and .npy files.")

        epochs_all: List[mne.Epochs] = []
        X_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []

        for fif_path, npy_path in zip(fif_files, npy_files):
            print(f"[EEGClassifier] Loading {fif_path.name} / {npy_path.name}")

            epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
            y = epochs.events[:, -1]

            features = np.load(npy_path).reshape(len(y), -1)  # flatten

            epochs_all.append(epochs)
            X_all.append(features)
            y_all.append(y)

        epochs_concat = mne.concatenate_epochs(epochs_all, verbose=False)
        X_concat = np.concatenate(X_all)
        y_concat = np.concatenate(y_all)

        print(f"Feature matrix: {X_concat.shape}, labels: {y_concat.shape}")

        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95),
            LinearDiscriminantAnalysis(),
        )

        # Guard against degenerate datasets where all features are constant.
        if np.allclose(X_concat, X_concat[0]):
            raise ValueError(
                "Training data has no variance; feature extraction might have failed."
            )

        try:
            pipe.fit(X_concat, y_concat)
        except IndexError as err:
            # scikit-learn can throw an obscure IndexError when PCA yields
            # zero components (e.g. if the variance is zero).  Provide a more
            # helpful message for the user.
            raise ValueError(
                "Classifier training failed: check that your recorded data contains\n"
                "valid, non-zero features."
            ) from err

        inv_op = create_inverse_operator(epochs_concat.info)
        return cls(pipe, inverse_operator=inv_op)

    @classmethod
    def make_giga_classifier(
        cls, X: np.ndarray, Y: np.ndarray
    ) -> Tuple["EEGClassifier", np.ndarray, np.ndarray]:
        """
        Train on a generic (X, Y) dataset, return the classifier + held-out test set.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95),
            LinearDiscriminantAnalysis(),
        )
        pipe.fit(X_train, y_train)
        return cls(pipe), X_test, y_test

    # --------------------------------------------------------------------- #
    #  Inference                                                             #
    # --------------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions for X."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for X."""
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError("Pipeline does not support probability estimates")
        return self.pipeline.predict_proba(X)

    # --------------------------------------------------------------------- #
    #  Reporting                                                             #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _print_results(pred: np.ndarray, y_true: np.ndarray) -> None:
        """Pretty print a small confusion overview in the console."""
        correct = int((pred == y_true).sum())
        accuracy = correct / len(pred)
        print("\nResults")
        print("-" * 30)
        for p, y in zip(pred, y_true):
            print(f"{p:<15}{y:<15}")
        print("\n" + "-" * 30)
        print(f"Accuracy: {accuracy:.2%} ({correct}/{len(pred)})")

    def save_test_results(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        subject_number: int,
        print_console: bool = True,
    ) -> None:
        """
        Persist test metrics in JSON (appended run-after-run) and optionally
        print a summary in the console.
        """
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if y_true.ndim > 1:
            y_true = y_true.flatten()

        if print_console:
            self._print_results(predictions, y_true)

        folder = Path(f"data/s{subject_number:02}")
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / "test_results.json"

        # Metrics
        accuracy = accuracy_score(y_true, predictions)
        right_recall = recall_score(y_true, predictions, zero_division=0)
        precision = precision_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        left_recall = tn / (tn + fp) if (tn + fp) else 0.0
        score = int((predictions == y_true).sum())

        # Load / update JSON
        if file_path.exists():
            with file_path.open("r") as fp:
                try:
                    results = json.load(fp)
                except json.JSONDecodeError:
                    results = {}
        else:
            results = {}

        test_key = f"test_{len(results) + 1}"
        results[test_key] = {
            "score": score,
            "predictions": predictions.tolist(),
            "y_true": y_true.tolist(),
            "accuracy": accuracy,
            "right_recall": right_recall,
            "left_recall": left_recall,
            "f1": f1,
            "precision": precision,
        }

        with file_path.open("w") as fp:
            json.dump(results, fp, indent=4)

        print(f"[EEGClassifier] Saved {test_key} results to {file_path}")


# --------------------------------------------------------------------------- #
#                               Public exports                                #
# --------------------------------------------------------------------------- #
__all__ = ["EEGClassifier", "check_for_existing_training_data"]