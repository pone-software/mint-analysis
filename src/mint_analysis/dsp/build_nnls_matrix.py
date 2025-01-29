"""
This module contains helper functions for converting template functions into matrices for nnls fitting.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import hankel
from lgdo.types import WaveformTable

def build_nnls_matrix(
    wf_template: WaveformTable, wf_len: int, wf_sampling: int
) -> tuple[np.ndarray, np.ndarray]:

    """
    This function builds the matrices needed for nnls fitting of a template to a waveform.
    It returns 2 matrices the first is the A matrix that is used in the nnls fitting 
    (with the downsampled template) and the second is the full A matrix that is used 
    to upsample the template.

    Parameters
    ----------
    wf_template : WaveformTable
        The waveform template to fit to the data.
    wf_len : int
        The length of the waveform to fit to.
    downsample_factor : int
        The factor by which to downsample the template.
    
    Returns
    -------
    A : np.ndarray
        The A matrix used in the nnls fitting.
    A_upsampled : np.ndarray
        The full A matrix used to upsample the template.
    """

    template_sampling = wf_template.dt.nda[0][0]
    upsample_factor = template_sampling / wf_sampling

    if upsample_factor != int(upsample_factor):
        raise ValueError("Template sampling rate is not a multiple of the waveform sampling rate")

    out_len = wf_len * upsample_factor

    padded_template = np.zeros(out_len)
    padded_template[: len(wf_template.values.nda[0])] = wf_template.values.nda[0]

    # Initialize A matrix , is it worth constraining so full template must be present?
    A_upsampled = hankel(padded_template[::-1], np.full(len(padded_template), 0))[::-1, :]
    A_upsampled = A_upsampled.astype("float32")

    sampling = np.zeros(out_len, dtype=bool)
    sampling[np.arange(0, out_len, upsample_factor, dtype=int)] = True
    A = A_upsampled[sampling]

    return A, A_upsampled
