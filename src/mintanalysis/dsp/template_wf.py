"""
This module contains functions for building template waveforms from calibration data.
They should take in a WaveformTable and return a WaveformTable with a single waveform.
"""

from __future__ import annotations

import numpy as np
from lgdo.types import ArrayOfEqualSizedArrays, WaveformTable


def build_template_waveform(
    wfs_in: WaveformTable, ts: np.ndarray, upsample_factor: int = 5
) -> WaveformTable:
    """
    This function builds an upsampled template waveform from a set of waveforms
    by aligning them at a given time.
    The waveforms are upsampled by a factor of `upsample_factor`.

    Parameters
    ----------

    wfs_in : WaveformTable
        The input waveforms to build the template.
    ts : np.ndarray
        The starting time of the waveforms used for aligning.
    upsample_factor : int
        The factor by which to upsample the waveforms.

    Returns
    -------
    wf_template : WaveformTable
        The upsampled template waveform.
    """

    if not isinstance(upsample_factor, int):
        msg = "Upsample factor must be an integer"
        raise ValueError(msg)

    # Initialize extended arrays
    wf_len = len(wfs_in.values.nda[0])
    wf_extended = np.zeros(wf_len * upsample_factor)
    counts = np.zeros(wf_len * upsample_factor)
    base = np.arange(wf_len)

    t0_shift = int(wf_len) / 4
    shifts_list = np.floor((ts - t0_shift) * upsample_factor).astype(int)

    shifts = shifts_list[:, None]  # Shape (num_waveforms, 1) for broadcasting
    indices = base * upsample_factor - shifts  # Shape (num_waveforms, wf_len)

    # Scatter-add values to wf_extended and counts
    for i in range(len(wfs_in)):
        wf_extended[indices[i]] += wfs_in.values.nda[i, :]
        counts[indices[i]] += np.ones(wf_len)

    start = int(t0_shift * upsample_factor)
    start_value = (wf_extended / counts)[start]
    end = int(
        np.ceil(
            (np.where((wf_extended / counts)[start:] < start_value)[0][0] + start)
            / upsample_factor
        )
        * upsample_factor
    )

    return WaveformTable(
        values=[ArrayOfEqualSizedArrays(nda=wf_extended[start:end])],
        dt=wfs_in.dt.nda[0][0] / upsample_factor,
        t0=0,
        t0_units=wfs_in.t0_units,
    )
