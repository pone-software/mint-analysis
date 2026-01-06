#!/usr/bin/env python3
"""
funny_heartbeat.py

Displays absurd waveform-to-physics heartbeat lines in the terminal.
- Escalates absurdity over runtime
- Occasionally prints rare easter egg lines
"""

import random
import threading
import time
from contextlib import contextmanager
import shutil
import sys

# --- core word lists ---
VERBS = [
    "Gently coercing",
    "Politely requesting",
    "Forcing",
    "Convincing",
    "Bribing",
    "Strongly encouraging",
    "Shaming",
    "Reassuring",
    "Whispering to",
    "Negotiating with",
    "Taming",
    "Pleasing",
]

OBJECTS = [
    "the low-resolution PMT trace",
    "these stubborn pulses",
    "the mischievous waveform",
    "the noisy PMT bins",
    "this opinionated signal",
    "the reluctant photon counts",
    "the wiggly PMT output",
    "the latent pulses",
    "the under-sampled peaks",
]

PHYSICS = [
    "for its hidden amplitudes",
    "to recover true photon arrival times",
    "for non-negative signal components",
    "to unfold underlying pulses",
    "for latent photon distribution",
    "to reconstruct the high-resolution waveform",
    "to infer pulse heights",
    "for all scientifically plausible signal shapes",
]

METHODS = [
    "using NNLS unfolding",
    "via repeated non-negative least squares",
    "by iterative pulse nudging",
    "with gentle spline persuasion",
    "through spectral telepathy",
    "by brute-force amplitude whispering",
    "using photon-level negotiations",
    "via zero-error curve hugging",
    "with mild FFT intimidation",
]

ENDINGS = [
    "…",
    "please cooperate…",
    "results are approximate™…",
    "the photons seem skeptical…",
    "stand by for high-resolution enlightenment…",
    "this might break reality…",
    "science is optional…",
    "the waveform resists all logic…",
]

# Rare “easter egg” lines
EASTER_EGGS = [
    "🎉 The PMTs are throwing a photon party!",
    "🛸 Aliens are tweaking your NNLS bins.",
    "🧙‍♂️ Wizard applied magic to the unfolding.",
    "💥 Chaos ensues in the photon distribution!",
    "🐱 Cats are supervising the waveform reconstruction.",
    "🌈 Rainbow pulses detected!",
    "🪄 Unfolding complete: miracles observed.",
]

SPINNER = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

# --- generate a line with escalating absurdity ---
def funny_line(elapsed_seconds: float) -> str:
    """Return an absurd waveform-to-physics line, escalation based on elapsed time."""
    # Escalate absurdity by expanding METHODS and ENDINGS over time
    extra_methods = [
        "while loudly chanting to the PMTs",
        "by negotiating with imaginary photons",
        "through time-traveling pulse inference"
    ]
    extra_endings = [
        "prepare for interdimensional interference…",
        "physics may be optional here…",
        "please wait while reality stabilizes…"
    ]

    methods_pool = METHODS + (extra_methods if elapsed_seconds > 60 else [])
    endings_pool = ENDINGS + (extra_endings if elapsed_seconds > 120 else [])

    line = f"{random.choice(VERBS)} {random.choice(OBJECTS)} {random.choice(PHYSICS)} ({random.choice(methods_pool)}). {random.choice(endings_pool)}"

    # Very rare easter egg (1% chance)
    if random.random() < 0.01:
        line = random.choice(EASTER_EGGS)

    return line

# --- heartbeat thread ---
def funny_heartbeat(stop_event, line_interval: float = 10.0, spinner_interval: float = 0.1):
    start_time = time.time()
    last_line_time = 0
    current_line = funny_line(0)
    spinner_idx = 0

    while not stop_event.is_set():
        elapsed = time.time() - start_time

        # Update the line every line_interval
        if elapsed - last_line_time >= line_interval:
            current_line = funny_line(elapsed)
            last_line_time = elapsed

        # Update spinner
        spinner_char = SPINNER[spinner_idx % len(SPINNER)]
        spinner_idx += 1

         # Get terminal width
        width = shutil.get_terminal_size((120, 20)).columns

        # Prepare full text and pad to clear old content
        full_text = f"{spinner_char} {current_line}"
        padded_text = full_text[:width-1].ljust(width-1)  # ensure fits terminal

        sys.stdout.write("\r" + padded_text)
        sys.stdout.flush()
        time.sleep(spinner_interval)

    # Clear line on exit
    sys.stdout.write("\r" + " " * 120 + "\r")
    sys.stdout.flush()


# --- context manager for easy use ---
@contextmanager
def heartbeat_ctx(line_interval: float = 10.0, spinner_interval: float = 0.1):
    stop = threading.Event()
    t = threading.Thread(target=funny_heartbeat, args=(stop, line_interval, spinner_interval), daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()

if __name__ == "__main__":

    with heartbeat_ctx():
        time.sleep(300)