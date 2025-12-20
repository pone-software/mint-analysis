import argparse
import json
import logging
import os

import numpy as np
from dspeed import build_dsp
from lgdo import lh5


def replace_list_with_array(dic: dict):
    """
    Recursively converts all lists found in a given dictionary to numpy arrays

    Parameters
    ----------
    dic : dict
        Dictionary to alter

    Returns
    -------
    dict
        Dictionary with all lists converted to numpy arrays
    """
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = replace_list_with_array(value)
        elif isinstance(value, list):
            dic[key] = np.array(value, dtype="float64")
        else:
            pass
    return dic


def build_dsp_cli():
    parser = argparse.ArgumentParser(description="Build DSP tier from RAW input.")
    parser.add_argument("-r", "--f_raw", help="Path to raw file", required=True)
    parser.add_argument(
        "-d",
        "--f_dsp",
        default=None,
        help="Path to raw file (if omitted replaces all occurrences of raw in f_raw with dsp)",
    )
    parser.add_argument("-c", "--f_config", help="Path to DSP config file")
    parser.add_argument(
        "-e",
        "--f_channel_config",
        default=None,
        help="Path to DSP channel config file (f_config becomes fall back)",
    )
    parser.add_argument("-p", "--f_db", help="Path to database file", required=True)
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Override existing output file"
    )

    args = parser.parse_args()
    f_dsp = args.f_raw.replace("raw", "dsp") if args.f_dsp is None else args.f_dsp
    # Create raw folders if not existing
    dir = os.path.dirname(f_dsp)
    if dir:
        os.makedirs(dir, exist_ok=True)

    logger = logging.getLogger("dspeed")
    log_level = logging.INFO
    logger.setLevel(log_level)

    fmt = logging.Formatter("[%(asctime)s] [%(name)s - %(funcName)s] [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_file = f_dsp.replace(f_dsp.split(".")[-1], "log")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    with open(args.f_db) as json_file:
        db_dic = json.load(json_file)

    db_dic = replace_list_with_array(db_dic)

    keys = lh5.ls(args.f_raw)
    if next(iter(db_dic.keys())) not in keys:
        db_dic = dict.fromkeys(keys, db_dic)
    build_dsp(
        raw_in=args.f_raw,
        dsp_out=f_dsp,
        dsp_config=args.f_config,
        database=db_dic,
        chan_config=args.f_channel_config,
        write_mode="r" if args.overwrite else None,
    )


if __name__ == "__main__":
    build_dsp_cli()
