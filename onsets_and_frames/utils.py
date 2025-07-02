import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from onsets_and_frames.constants import (
    DTW_FACTOR,
    HOP_LENGTH,
    MAX_MIDI,
    MIN_MIDI,
    N_KEYS,
)


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def shift_label(label, shift):
    if shift == 0:
        return label
    assert len(label.shape) == 2
    t, p = label.shape
    keys, instruments = N_KEYS, p // N_KEYS
    label_zero_pad = torch.zeros(t, instruments, abs(shift), dtype=label.dtype)
    label = label.reshape(t, instruments, keys)
    to_cat = (
        (label_zero_pad, label[:, :, :-shift])
        if shift > 0
        else (label[:, :, -shift:], label_zero_pad)
    )
    label = torch.cat(to_cat, dim=-1)
    return label.reshape(t, p)


def get_peaks(notes, win_size, gpu=False):
    constraints = []
    notes = notes.cpu()
    for i in range(1, win_size + 1):
        forward = torch.roll(notes, i, 0)
        forward[:i, ...] = 0  # assume time axis is 0
        backward = torch.roll(notes, -i, 0)
        backward[-i:, ...] = 0
        constraints.extend([forward, backward])
    res = torch.ones(notes.shape, dtype=bool)
    for elem in constraints:
        res = res & (notes >= elem)
    return res if not gpu else res.cuda()


def get_peaks_numpy(notes, win_size):
    """
    Detect peaks in a NumPy array based on a window size.

    Args:
        notes (np.ndarray): Input array, shape (frames, ...).
        win_size (int): Window size for detecting peaks.

    Returns:
        np.ndarray: Boolean array indicating peaks, same shape as `notes`.
    """
    # Initialize constraints
    constraints = []
    notes = np.array(notes)  # Ensure input is a NumPy array

    for i in range(1, win_size + 1):
        # Roll array forward and backward
        forward = np.roll(notes, i, axis=0)
        backward = np.roll(notes, -i, axis=0)

        # Zero out invalid regions
        forward[:i, ...] = 0
        backward[-i:, ...] = 0

        constraints.extend([forward, backward])

    # Initialize result with all True
    res = np.ones_like(notes, dtype=bool)

    # Apply constraints
    for elem in constraints:
        res &= notes >= elem

    return res


def get_diff(notes, offset=True):
    rolled = np.roll(notes, 1, axis=0)
    rolled[0, ...] = 0
    return (rolled & (~notes)) if offset else (notes & (~rolled))


def compress_across_octave(notes):
    keys = MAX_MIDI - MIN_MIDI + 1
    time, instruments = notes.shape[0], notes.shape[1] // keys
    notes_reshaped = notes.reshape((time, instruments, keys))
    notes_reshaped = notes_reshaped.max(axis=1)
    octaves = keys // 12
    res = np.zeros((time, 12), dtype=np.uint8)
    for i in range(octaves):
        curr_octave = notes_reshaped[:, i * 12 : (i + 1) * 12]
        res = np.maximum(res, curr_octave)
    return res


def compress_time(notes, factor):
    t, p = notes.shape
    res = np.zeros((t // factor, p), dtype=notes.dtype)
    for i in range(t // factor):
        res[i, :] = notes[i * factor : (i + 1) * factor, :].max(axis=0)
    return res


def get_matches(index1, index2):
    matches = {}
    for i1, i2 in zip(index1, index2):
        # matches[i1] = matches.get(i1, []) + [i2]
        if i1 not in matches:
            matches[i1] = []
        matches[i1].append(i2)
    return matches


"""
Extend a temporal range to WINDOW_SIZE_SRC if it is shorter than that.
WINDOW_SIZE_SRC defaults to 28 frames for 256 hop length (assuming DTW_FACTOR=3), which is ~0.5 second.
"""


def get_margin(
    t_sources, max_len, WINDOW_SIZE_SRC=11 * (512 // HOP_LENGTH) + 2 * DTW_FACTOR
):
    margin = max(0, (WINDOW_SIZE_SRC - len(t_sources)) // 2)
    t_sources_left = list(range(max(t_sources[0] - margin, 0), t_sources[0]))
    t_sources_right = list(
        range(t_sources[-1], min(t_sources[-1] + margin, max_len - 1))
    )
    t_sources_extended = t_sources_left + t_sources + t_sources_right
    return t_sources_extended


def get_inactive_instruments(target_onsets, T):
    keys = MAX_MIDI - MIN_MIDI + 1
    time, instruments = target_onsets.shape[0], target_onsets.shape[1] // keys
    notes_reshaped = target_onsets.reshape((time, instruments, keys))
    active_instruments = notes_reshaped.max(axis=(0, 2))
    res = np.zeros((T, instruments, keys), dtype=bool)
    for ins in range(instruments):
        if active_instruments[ins] == 0:
            res[:, ins, :] = 1
    return res.reshape((T, instruments * keys)), active_instruments


def max_inst(probs, threshold_vec=None):
    if threshold_vec is None:
        threshold_vec = 0.5
    if probs.shape[-1] == N_KEYS or probs.shape[-1] == N_KEYS * 2:
        # there is only pitch
        return probs
    keys = MAX_MIDI - MIN_MIDI + 1
    instruments = probs.shape[1] // keys
    time = len(probs)
    probs = probs.reshape((time, instruments, keys))
    notes = probs.max(axis=1) >= threshold_vec
    max_instruments = np.argmax(probs[:, :-1, :], axis=1)
    res = np.zeros(probs.shape, dtype=np.uint8)
    for t, p in zip(*(notes.nonzero())):
        res[t, max_instruments[t, p], p] = 1
        res[t, -1, p] = 1
    return res.reshape((time, instruments * keys))


# Define the smoothing function (operates on CPU)
def smooth_labels(onset_tensor):
    """
    Smooths onset labels using a triangular kernel with 1D convolution along the time axis.

    Args:
        onset_tensor (torch.Tensor): A (T, F) tensor where T = time steps and F = pitches.

    Returns:
        torch.Tensor: Smoothed onset tensor with the same shape (T, F).
    """
    # Define the triangular smoothing kernel
    # kernel = torch.tensor([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2],
    #                       dtype=onset_tensor.dtype).view(1, 1, -1)
    # kernel = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5,  0.6, 0.7,  0.8, 0.9,  1, 0.9,  0.8, 0.7,  0.6, 0.5,  0.4, 0.3,  0.2, 0.1],
    #                       dtype=onset_tensor.dtype).view(1, 1, -1)
    kernel = torch.tensor([0.33, 0.67, 1, 0.67, 0.33], dtype=onset_tensor.dtype).view(
        1, 1, -1
    )

    onset_tensor = onset_tensor.T.unsqueeze(1)  # Now shape is (F, 1, T)

    # Use 'same' padding so that the output has the same time dimension as the input.
    padding = kernel.shape[-1] // 2
    smoothed = F.conv1d(onset_tensor, kernel, padding=padding)

    # Reshape back to original shape (T, F)
    return smoothed.squeeze(1).T


def initialize_logging_system(logdir):
    """Initialize the logging system once with named loggers for train and dataset."""
    log_file = os.path.join(logdir, "training.log")

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler (shared by all loggers)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler (shared by all loggers)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Create train logger
    train_logger = logging.getLogger("train")
    train_logger.setLevel(logging.INFO)
    train_logger.handlers.clear()
    train_logger.addHandler(file_handler)
    train_logger.addHandler(console_handler)

    # Create dataset logger
    dataset_logger = logging.getLogger("dataset")
    dataset_logger.setLevel(logging.INFO)
    dataset_logger.handlers.clear()
    dataset_logger.addHandler(file_handler)
    dataset_logger.addHandler(console_handler)

    return train_logger, dataset_logger


def get_logger(name):
    """Get a named logger. Call initialize_logging_system first."""
    return logging.getLogger(name)
