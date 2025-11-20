import argparse
import json

import numpy as np
from dspeed import build_dsp


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple example script.")
    parser.add_argument("-r", "--f_raw", help="Path to raw file", required=True)
    parser.add_argument("-c", "--f_config", help="Path to DSP config file", required=True)
    parser.add_argument("-d", "--f_db", help="Path to database file", required=True)
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Override existing output file"
    )

    args = parser.parse_args()
    f_dsp = args.f_raw.replace("raw", "dsp")

    with open(args.f_config) as json_file:
        config = json.load(json_file)

    ch_dic = {f"ch{i:03}": config for i in range(8)}

    with open(args.f_db) as json_file:
        db_dic = json.load(json_file)

    db_dic = replace_list_with_array(db_dic)
    db_dic = {f"ch{i:03}": db_dic for i in range(8)}
    build_dsp(
        f_raw=args.f_raw,
        f_dsp=f_dsp,
        database=db_dic,
        chan_config=ch_dic,
        write_mode="r" if args.overwrite else None,
    )
