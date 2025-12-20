"""
Helper functions for ana tier
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from uncertainties import UFloat, ufloat


def setup_logging(log_file: Path = "analysis.log", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("PESpectrum")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] [%(name)s - %(funcName)s] [%(levelname)s] %(message)s"
        )

        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_physics_object(obj, ureg):
    """
    Convert a given object recursevly
    to a object with all {val,(err),unit} occurrences replaced with a pint (and ufloat) object
    """
    if isinstance(obj, dict) and {"val", "unit"}.issubset(obj):
        val = obj["val"]
        if "err" in obj:
            err = obj["err"]
            return ureg.Quantity(ufloat(val, err), ureg(obj["unit"]))
        return ureg.Quantity(val, ureg(obj["unit"]))

    if isinstance(obj, dict):
        return {k: get_physics_object(v, ureg) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(get_physics_object(v, ureg) for v in obj)

    return obj


def quantity_to_dict(obj, val_key="val", err_key="err", unit_key="unit"):
    """
    Recursively replace a pint (and ufloat) object with a dict
    {val, (err), unit}
    """
    # The pint internal function is broken (see issue 2121)
    preferred_units = ["Hz"]
    if hasattr(obj, "magnitude") and hasattr(obj, "units"):
        if preferred_units is not None:
            for u in preferred_units:
                if obj.is_compatible_with(u):
                    obj = obj.to(u)
        mag = obj.magnitude

        if isinstance(mag, UFloat):
            val = mag.n
            err = mag.s
        else:
            val = mag
            err = None

        return {
            val_key: float(val),
            **({err_key: float(err)} if err is not None else {}),
            unit_key: (
                str(obj.units)
                if obj.units.is_compatible_with("ohm") or obj.units.is_compatible_with("degC")
                else format(obj.units, "~")
            ),
        }

    if isinstance(obj, dict):
        return {k: quantity_to_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(quantity_to_dict(v) for v in obj)

    return obj


def linear_func(x, a, b):
    return a * x + b


def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def poisson_nll(amp: float, mu: float, sigma: float, x: np.ndarray, y: np.ndarray) -> float:
    expected = gaussian(x, amp, mu, sigma)
    expected = np.clip(expected, 1e-10, None)
    return float(np.sum(expected - y * np.log(expected)))


def valley_index_strict(y: np.ndarray) -> tuple[int, int] | None:
    """Return (peak_idx, valley_idx) or None."""
    n = len(y)
    if n < 3:
        return None

    peak = None
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peak = i
            break
    if peak is None:
        return None

    valley = peak
    i = peak + 1
    while i < n:
        if y[i] > y[i - 1]:
            return peak, valley
        if y[i] < y[valley]:
            valley = i
        i += 1

    return None
