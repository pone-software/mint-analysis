import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit


def linear_func(x, a, b):
    return a * x + b


if __name__ == "__main__":

    f_result = "/home/pkrause/software/mint-analysis/debug_out/results.yaml"
    with open(f_result) as file:
        result_dic = yaml.safe_load(file)

    out_folder = "/home/pkrause/software/mint-analysis/debug_out/gain/"

    bin_size = 20
    bins = np.arange(-100, 10000, bin_size)
    lim = 20
    A4_LANDSCAPE = (11.69, 8.27)

    override_results = True

    if os.path.exists(f_result):
        with open(f_result) as file:
            result_dic = yaml.safe_load(file)
    else:
        raise RuntimeError("File does not exist")

    if "pe_spectrum" not in result_dic:
        raise RuntimeError("pe_spectrum info not present!")

    if "linear_gain" in result_dic and not override_results:
        raise RuntimeError("results already exist and override flag not set")

    ######################
    # Gain calculation   #
    ######################
    tmp_dic = {"used_keys": []}
    y_unit = None
    x_unit = None
    for key, run in result_dic["pe_spectrum"].items():
        tmp_dic["used_keys"].append(key)
        for pmt in run:
            if pmt not in tmp_dic:
                tmp_dic[pmt] = {"voltage": [], "vals": [], "errs": []}
            v = run[pmt]["voltage"]["val"]
            xu = run[pmt]["voltage"]["unit"]
            if x_unit is None:
                x_unit = xu
            else:
                assert x_unit == xu
            if v == 0:
                continue

            tmp_dic[pmt]["voltage"].append(v)
            tmp_dic[pmt]["vals"].append(run[pmt]["pe_peak_fit"]["mean"]["val"])
            tmp_dic[pmt]["errs"].append(run[pmt]["pe_peak_fit"]["mean"]["err"])

            yu = run[pmt]["pe_peak_fit"]["mean"]["unit"]
            if y_unit is None:
                y_unit = yu
            else:
                assert y_unit == yu

    with PdfPages(out_folder + "gain_plots.pdf") as pdf:
        for key, pmt in tmp_dic.items():
            if key == "used_keys":
                continue
            fig, ax = plt.subplots()
            fig.set_figwidth(A4_LANDSCAPE[0])
            fig.set_figheight(A4_LANDSCAPE[1])
            ax.errorbar(
                pmt["voltage"],
                pmt["vals"],
                pmt["errs"],
                label=f"PMT {key}",
                fmt="o",
            )
            ax.set_ylabel(f"PMT position ({y_unit})")
            ax.set_xlabel(f"Voltage ({x_unit})")

            params, covariance = curve_fit(
                linear_func,
                pmt["voltage"],
                pmt["vals"],
                sigma=pmt["errs"],
                absolute_sigma=True,
            )
            a_opt, b_opt = params
            perr = np.sqrt(np.diag(covariance))
            x = np.linspace(-1 * b_opt / a_opt, 110, 1000)
            ax.plot(x, linear_func(x, a_opt, b_opt), ls="--", color="red", label="Fit")

            tmp_dic[key]["a"] = {
                "val": float(a_opt),
                "err": float(perr[0]),
                "unit": f"{y_unit}/{x_unit}",
            }
            tmp_dic[key]["b"] = {
                "val": float(b_opt),
                "err": float(perr[1]),
                "unit": f"{y_unit}/{x_unit}",
            }
            tmp_dic[key]["func"] = "G = a*voltage+b"

            pmt.pop("errs")
            pmt.pop("vals")
            pmt.pop("voltage")
            ax.legend()
            pdf.savefig()
            plt.close()

    result_dic["linear_gain"] = tmp_dic


with open(f_result, "w") as file:
    yaml.safe_dump(result_dic, file, default_flow_style=False)
