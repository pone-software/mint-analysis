import argparse
import json
import os

import numpy as np
from lgdo import lh5


def gumbel_pdf(x: np.ndarray, mu: float, sigma: float):
    """
    Create a Gumbel PDF (https://en.wikipedia.org/wiki/Gumbel_distribution)

    Parameters
    ----------
    x : ndarray
        Array of x values along which to create the function.
    mu : float
        Mean position of the Gumble PDF.
    sigma : float
        Width of the Gumble PDF in sigma.

    Returns
    -------
    ndarray
        y values of the Gumbel PDF.
    """
    beta = sigma * (np.sqrt(6) / np.pi)
    z = (x - mu) / beta
    return (1 / beta) * np.exp(-(z + np.exp(-1 * z)))


def downsample(x: np.ndarray, N: int):
    """
    Downsample a given array by taking means.

    Parameters
    ----------
    x : ndarray
        Array of x values along which to create the function.
    N : int
        Downsample factor

    Returns
    -------
    ndarray
        downsampled array of x
    """
    return np.append(x, np.zeros(len(x) % N) * np.nan).reshape(-1, N).mean(axis=1)


def create_nnls_database(
    f_out: str,
    f_raw: str,
    daq_sampling: float = 4.8e-9,
    upsampling_factor: int = 10,
    gumble_sigma: float = 4e-9,
    folding_func=None,
    *args,
    **kwargs,
):
    """
    Produce a NNLS matrix database file for the DSP processing

    Parameters
    ----------
    f_out : str
        Output path of the NNLS matrix database file
    f_raw : str
        Path to a reference raw file with waveforms. Must contain ch000/raw/waveform/values
    daq_sampling: float
        Sampling time of the DAQ in seconds
    upsampling_factor: int
        Upsampling factor of the goal matrix
    gumble_sigma: float
        Width of the Gumble PDF in sigma.
    folding_func:
        Optional: Pass additional folding method

    """
    data = lh5.read_as("ch000/raw", f_raw, "ak")
    x = np.arange(0, len(data.waveform.values[0]) * daq_sampling, daq_sampling / upsampling_factor)
    A = np.zeros((len(data.waveform.values[0]), len(x)))
    for i in range(len(x)):
        if folding_func:
            A[:, i] = downsample(
                folding_func(gumbel_pdf(x, x[i], gumble_sigma), *args, **kwargs), upsampling_factor
            )
        else:
            A[:, i] = downsample(gumbel_pdf(x, x[i], gumble_sigma), upsampling_factor)

    db_dic = {
        "gumbel": {
            "sigma_in_ns": gumble_sigma,
            "right_hand_vector_length": A.shape[0],
            "right_hand_vector_resolution_in_ns": daq_sampling,
            "solution_vector_length": A.shape[1],
            "solution_vector_resolution_in_ns": daq_sampling / upsampling_factor,
            "template function": "gumbel pdf",
            "matrix": A.tolist(),
        }
    }

    with open(f_out, "w") as outfile:
        json.dump(db_dic, outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple example script.")
    parser.add_argument("-r", "--f_raw", help="Path to raw file", required=True)
    parser.add_argument("-d", "--f_db", help="Path to database file", required=True)
    parser.add_argument("-d", "--f_db", help="Path to database file", required=True)
    parser.add_argument(
        "-u",
        "--upsampling_factor",
        default=10,
        help="Upsampling factor relative to DAQ sampling time",
    )
    parser.add_argument("-s", "--sigma", default=4, help="Sigma of the gumble function in ns")
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Override existing output file"
    )

    args = parser.parse_args()

    if os.path.exists(args.f_db) and not args.overwrite:
        msg = "NNLS database already exists!"
        raise ValueError(msg)

    create_nnls_database(
        args.f_db,
        args.f_raw,
        upsampling_factor=args.upsampling_factor,
        gumble_sigma=args.sigma * 1e-9,
    )
