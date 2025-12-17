import numpy as np
import pytest
from lgdo.types import ArrayOfEqualSizedArrays, WaveformTable

from mintanalysis.pmt.dsp.template_wf import build_template_waveform


def test_build_template_waveform():
    # Create a mock WaveformTable
    wfs_in_values = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
    wfs_in = WaveformTable(
        values=ArrayOfEqualSizedArrays(nda=wfs_in_values), dt=[1, 1, 1, 1], t0=0, t0_units="ns"
    )
    ts = np.array([1.5, 2.0, 1.0, 2.5])
    upsample_factor = 2

    # Call the function
    wf_template = build_template_waveform(wfs_in, ts, upsample_factor)

    # Check the output type
    assert isinstance(wf_template, WaveformTable)

    # Check the shape of the output waveform
    expected_length = len(wfs_in_values[0]) * upsample_factor - 2
    assert len(wf_template.values.nda[0]) == expected_length

    # Check the dt value
    assert wf_template.dt[0] == wfs_in.dt[0] / upsample_factor


def test_build_template_waveform_invalid_upsample_factor():
    # Create a mock WaveformTable
    wfs_in_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    wfs_in = WaveformTable(
        values=ArrayOfEqualSizedArrays(nda=wfs_in_values), dt=[1, 1, 1], t0=0, t0_units="ns"
    )
    ts = np.array([0.5, 1.0, 1.5])
    upsample_factor = 2.5  # Invalid upsample factor

    # Check for ValueError
    with pytest.raises(ValueError, match="Upsample factor must be an integer"):
        build_template_waveform(wfs_in, ts, upsample_factor)
