# MISC
import glob
import importlib.util

import numpy as np

# LEGEND specific imports
from dspeed import build_dsp
from lgdo import ArrayOfEqualSizedArrays, Table, WaveformTable, lh5

# Load the .py config as a module
config_file = "dspeed_config_PONE_1.py"
spec = importlib.util.spec_from_file_location("config", config_file)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
# Access the config dictionary
dsp_config = config_module.config


# Functions
def build_raw(RAW_PATH: str, lh5_file: str, signal: dict):
    """
    Gets data converted to lh5 and dumps it into Pygama format.

    Parameters:
        RAW_PATH: str - Path to raw converted file.
        lh5_file: str - Path to lh5 converted file.
        signal:   dict - Dictionary of stored channel data.

    Returns:
        f_raw : table - Output of Pygama converted file.
    """
    f_raw = RAW_PATH + lh5_file
    for k in signal.items():
        a = ArrayOfEqualSizedArrays(nda=np.array(signal[k], dtype=np.uint16))
        b = WaveformTable(values=a, dt=5, dt_units="ns", t0=0, t0_units="ns")
        # add everything into a lh5 structure and write it to disk
        table = Table(size=len(b))
        table.add_field("waveform", b, True)
        lh5.write(table, name="raw", group=k, lh5_file=f_raw)

    return f_raw


def database(
    A: np.ndarray, A_upsampled: np.ndarray, out_len: int, channel_name_1: str, channel_name_2: str
):
    """
    Database storing values of the matrices for nnls processor.

    Parameters:
        A:              np.ndarray - Base matrix (m,n).
        A_upsampled:    np.ndarray - Upsampled Matrix (n,n).
        out_len:        int - Length of out vector (n).
        channel_name_1: str - name of your channel 1.
        channel_name_2: str - name of your channel 2.

    Returns:
        f_dsp : table - Output of dspeed converted file.
    """
    return {
        channel_name_1: {
            "coefficient_matrix": A,
            "upsampled_matrix": A_upsampled.T,
            "solution_vector_length": out_len,
            "solution_vector_resolution_in_ns": 1,
        },
        channel_name_2: {
            "coefficient_matrix": A,
            "upsampled_matrix": A_upsampled.T,
            "solution_vector_length": out_len,
            "solution_vector_resolution_in_ns": 1,
        },
    }


def config_dsp(RAW_PATH: str, f_raw: dict, database: dict):
    """
    Configures your dspeed based on raw lh5 input.

    Parameters:
        RAW_PATH: str - Path to lh5 converted file.

    Returns:
        f_dsp : table - Output of dspeed converted file.
    """
    f_raw = glob.glob(RAW_PATH + "*")
    for f in f_raw:
        f_dsp = f.replace("raw", "dsp")
        build_dsp(
            f_raw=f,
            f_dsp=f_dsp,
            dsp_config=dsp_config,
            database=database,
            write_mode="o",
        )

    return f_dsp
