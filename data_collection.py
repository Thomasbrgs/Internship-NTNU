"""
data_collection.py – EEG acquisition helpers

* LSLCollector  → open / flush / collect windows from an LSL stream
* load_giga_data()  → convenience loader for the GIGA .mat recordings
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import mne
import numpy as np
import pylsl
import pymatreader

import constants as const 

# --------------------------------------------------------------------------- #
#                               LSL acquisition                               #
# --------------------------------------------------------------------------- #
class LSLCollector:
    """
    Handle a single LSL EEG stream.

    Example
    -------
    >>> collector = LSLCollector()            # auto-detect default stream
    >>> samples, ts = collector.collect_window()
    >>> print(samples.shape)                  # (n_channels, C.SAMPLE_WINDOW * sfreq)
    >>> discarded = collector.clear_buffer()  # empty remaining backlog
    """

    def __init__(self, stream_name: str | None = None) -> None:
        self.stream_name = stream_name or const.STREAM_NAME
        self.inlet: pylsl.StreamInlet = self._create_lsl_inlet(self.stream_name)
        self.info: mne.Info = self._create_mne_info(self.inlet)
        self.offset: float = self.inlet.time_correction()

    # --------------------------------------------------------------------- #
    # private helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _create_lsl_inlet(stream_name: str) -> pylsl.StreamInlet:
        """
        Return an inlet for the first LSL stream whose name == stream_name.
        Works with both pylsl APIs (resolve_stream / resolve_streams).
        """
        if hasattr(pylsl, "resolve_stream"):           # pylsl ≤ 1.16
            try:
                streams = pylsl.resolve_stream("name", stream_name, timeout=5)
            except TypeError:
                # older pylsl versions don't accept the timeout keyword
                streams = pylsl.resolve_stream("name", stream_name, 1, 5)
        else:                                          # pylsl ≥ 1.17
            # resolve_streams(wait_time=…) returns *all* streams → filter
            all_streams = pylsl.resolve_streams(wait_time=5.0)
            streams = [s for s in all_streams if s.name() == stream_name]

        if not streams:
            raise RuntimeError(f"No LSL stream named '{stream_name}' found.")

        inlet = pylsl.StreamInlet(streams[0])
        inlet.pull_sample()  # prime buffer
        return inlet
    
    @staticmethod
    def _create_mne_info(inlet: pylsl.StreamInlet) -> mne.Info:
        n_channels = inlet.info().channel_count()
        if n_channels == 4:
            ch_names = const.C.CH_NAMES_4
        elif n_channels == 8:
            ch_names = const.C.CH_NAMES_8
        elif n_channels == 32:
            ch_names = const.C.CH_NAMES_32
        elif n_channels == 64:
            ch_names = const.C.CH_NAMES_64
        else:
            raise ValueError(f"Unsupported channel count: {n_channels}")

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=inlet.info().nominal_srate(),
            ch_types="eeg",
        )
        info.set_montage(mne.channels.make_standard_montage("standard_1020"))
        return info

    @staticmethod
    def _timestep_correction(
        sample: np.ndarray,
        timestamps: np.ndarray,
        out_of_order: bool = True,
        dejitter: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sort timestamps and apply linear de-jitter if requested."""
        timestamps = timestamps - timestamps.min()

        if out_of_order:
            order = np.argsort(timestamps)
            sample = sample[:, order]
            timestamps = timestamps[order]

        if dejitter:
            timestamps = np.linspace(timestamps.min(), timestamps.max(), len(timestamps))
        return sample, timestamps

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def collect_window(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Pull one window of size C.SAMPLE_WINDOW (seconds).

        Returns
        -------
        sample : ndarray, shape (n_channels, n_samples)
        timestamps : ndarray, shape (n_samples,)
        """
        sfreq = self.info["sfreq"]
        n_samples = int(sfreq * const.C.SAMPLE_WINDOW)

        sample_arr = np.empty((len(self.info["ch_names"]), 0))
        ts_arr = np.empty(0)

        while sample_arr.shape[1] < n_samples:
            need = n_samples - sample_arr.shape[1]
            chunk, ts = self.inlet.pull_chunk(
                timeout=const.C.SAMPLE_WINDOW,
                max_samples=need,
            )
            if not chunk:
                break
            sample_arr = np.hstack((sample_arr, np.asarray(chunk).T))
            ts_arr = np.concatenate((ts_arr, np.asarray(ts) + self.offset))

        if sample_arr.shape[1] != n_samples:
            print(
                f"[LSLCollector] Warning: sample shape {sample_arr.shape} != "
                f"expected {(len(self.info['ch_names']), n_samples)}"
            )

        return self._timestep_correction(sample_arr, ts_arr)

    def clear_buffer(self) -> int:
        """Flush pending data and return the number of discarded samples."""
        sample, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=1_000_000)
        return len(sample)


# --------------------------------------------------------------------------- #
#                     Dummy collector for offline mode                        #
# --------------------------------------------------------------------------- #
class DummyCollector:
    """
    Stand-in when no LSL stream is available.
    Returns zeros with plausible dimensions so the game can keep running.
    """

    def __init__(self, n_channels: int = 8, sfreq: int = 200) -> None:
        self._sfreq = sfreq
        self._info = mne.create_info(
            ch_names=[f"Dummy{i}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types="eeg",
        )

    # match the API of LSLCollector
    @property
    def info(self) -> mne.Info:
        return self._info

    def collect_window(self) -> tuple[np.ndarray, np.ndarray]:
        n_samples = int(const.C.SAMPLE_WINDOW * self._sfreq)
        data = np.zeros((len(self._info["ch_names"]), n_samples))
        ts = np.linspace(0, const.C.SAMPLE_WINDOW, n_samples)
        return data, ts

    def clear_buffer(self) -> int:
        return 0

# --------------------------------------------------------------------------- #
#                     Loader for pre-recorded (GIGA) .mat files               #
# --------------------------------------------------------------------------- #
def load_giga_data(
    subject_number: int,
    set_average_reference: bool = True,
    filter: Tuple[int, int] | None = None,
    baseline: Tuple[float, float] | None = None,
    reject: dict | None = None,
    notch_filter: float | None = None,
    epochs_only: bool = True,
):
    """
    Read the GIGA dataset for a given subject and build MNE objects.

    Returns
    -------
    If epochs_only is True
        epochs_left, epochs_right
    else
        raw_left, raw_right, epochs_left, epochs_right
    """
    ch_types = ["eeg"] * 64
    mat_path = Path("giga_mat_files") / f"s{subject_number:02}.mat"
    rawdata = pymatreader.read_mat(str(mat_path))

    montage = mne.channels.make_standard_montage("biosemi64")
    info = (
        mne.create_info(
            const.C.CH_NAMES_64,
            rawdata["eeg"]["srate"],
            ch_types=ch_types,
            verbose=False,
        )
        .set_montage(montage)
    )
    info["subject_info"] = dict(id=subject_number, his_id=rawdata["eeg"]["subject"])

    mi_left = rawdata["eeg"]["imagery_left"]
    mi_right = rawdata["eeg"]["imagery_right"]
    raw_left = mne.io.RawArray(mi_left[:64] * 1e-8, info, verbose=False)
    raw_right = mne.io.RawArray(mi_right[:64] * 1e-8, info, verbose=False)

    # --- optional pre-processing --------------------------------------- #
    if set_average_reference:
        raw_left.set_eeg_reference(ref_channels="average", projection=True, verbose=False)
        raw_right.set_eeg_reference(ref_channels="average", projection=True, verbose=False)

    if filter:
        raw_left.filter(*filter, verbose=False)
        raw_right.filter(*filter, verbose=False)

    if notch_filter:
        raw_left.notch_filter(notch_filter, verbose=False)
        raw_right.notch_filter(notch_filter, verbose=False)

    # --- events & epochs ------------------------------------------------ #
    event_idx = [i for i, flag in enumerate(rawdata["eeg"]["imagery_event"]) if flag == 1]
    events_left = np.column_stack(
        (event_idx, np.zeros_like(event_idx), np.zeros_like(event_idx))
    )
    events_right = np.column_stack(
        (event_idx, np.zeros_like(event_idx), np.ones_like(event_idx))
    )

    tmin, tmax = -2.0, 5.0
    epochs_left = mne.Epochs(
        raw_left,
        events_left,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        event_id={"MI_left": 0},
        reject=reject,
        verbose=False,
    )
    epochs_right = mne.Epochs(
        raw_right,
        events_right,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        event_id={"MI_right": 1},
        reject=reject,
        verbose=False,
    )

    # --- drop bad trials ----------------------------------------------- #
    bad_left = [
        idx - 1
        for idx in (
            rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_mi"][0]
            + rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_voltage"][0]
        )
    ]
    bad_right = [
        idx - 1
        for idx in (
            rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_mi"][1]
            + rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_voltage"][1]
        )
    ]
    epochs_left.drop(bad_left, verbose=False)
    epochs_right.drop(bad_right, verbose=False)

    if epochs_only:
        return epochs_left, epochs_right
    return raw_left, raw_right, epochs_left, epochs_right