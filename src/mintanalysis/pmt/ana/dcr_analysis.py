import os

import yaml


def linear_func(x, a, b):
    return a * x + b


if __name__ == "__main__":

    f_result = "/home/pkrause/software/mint-analysis/debug_out/results.yaml"
    override_results = True
    time_mode = "aux"

    if os.path.exists(f_result):
        with open(f_result) as file:
            result_dic = yaml.safe_load(file)
    else:
        raise RuntimeError("File does not exist")

    if "pe_spectrum" not in result_dic.keys():
        raise RuntimeError("pe_spectrum info not present!")

    if "dcr" in result_dic and not override_results:
        raise RuntimeError("results already exist and override flag not set")
    result_dic["dcr"] = {}
    for rk, run in result_dic["pe_spectrum"].items():
        result_dic["dcr"][rk] = {}
        for key, pmt in run.items():
            if pmt["voltage"]["val"] == 0:
                continue
            if "dcr" in pmt and not override_results:
                print(
                    f"DCR key exists but override flag is not set! --> skipping {pmt} in run {run}"
                )
                continue
            dcts = (
                pmt["statistics"]["1st_pe_fit_integral_below_valley"]["val"]
                + pmt["statistics"]["cts_above_valley"]["val"]
            )
            result_dic["dcr"][rk][key] = {
                "val": dcts / pmt["runtime"][time_mode],
                "err": (dcts**0.5) / pmt["runtime"][time_mode],
                "unit": "Hz",
            }


with open(f_result, "w") as file:
    yaml.safe_dump(result_dic, file, default_flow_style=False)
