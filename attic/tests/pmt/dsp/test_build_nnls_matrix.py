import numpy as np
import pytest
from lgdo.types import ArrayOfEqualSizedArrays, WaveformTable
from mintanalysis.pmt.dsp.build_nnls_matrix import build_nnls_matrix


def test_build_nnls_matrix_valid_input():
    wf_template = WaveformTable(
        values=ArrayOfEqualSizedArrays(nda=[[1, 2, 3, 4, 5]]),
        dt=1,
        t0=0,
    )
    wf_len = 5
    wf_sampling = 2

    A, A_upsampled = build_nnls_matrix(wf_template, wf_len, wf_sampling)

    assert A.shape == (5, 10)
    assert A_upsampled.shape == (10, 10)
    assert np.allclose(A_upsampled[:, 1], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(A[:, 1], [0.0, 2.0, 4.0, 0.0, 0.0])


def test_build_nnls_matrix_invalid_upsample_factor():
    wf_template = WaveformTable(
        values=ArrayOfEqualSizedArrays(nda=[[1, 2, 3, 4, 5]]),
        dt=3,
        t0=0,
    )
    wf_len = 5
    wf_sampling = 2

    with pytest.raises(
        ValueError, match="Template sampling rate is not a multiple of the waveform sampling rate"
    ):
        build_nnls_matrix(wf_template, wf_len, wf_sampling)


def test_build_nnls_matrix_zero_padding():
    wf_template = WaveformTable(
        values=ArrayOfEqualSizedArrays(nda=[[1, 2, 3]]),
        dt=1,
        t0=0,
    )
    wf_len = 5
    wf_sampling = 1

    A, A_upsampled = build_nnls_matrix(wf_template, wf_len, wf_sampling)

    assert A.shape == (5, 5)
    assert A_upsampled.shape == (5, 5)
    assert np.allclose(A[2], [3, 2, 1, 0, 0])
    assert np.allclose(A_upsampled[2], [3, 2, 1, 0, 0])
