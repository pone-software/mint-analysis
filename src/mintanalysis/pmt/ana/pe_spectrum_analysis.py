import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import yaml
from iminuit import Minuit
from lgdo import lh5
from matplotlib.backends.backend_pdf import PdfPages


def valley_index_strict(y):
    """
    Return the index of the valley between the first strict peak
    and the next rise. Returns None if not found.
    """
    n = len(y)
    if n < 3:
        return None

    # 1. Find first strict peak
    peak = None
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peak = i
            break
    if peak is None:
        return None

    # 2. Walk downward and track the minimum
    valley = peak
    i = peak + 1

    while i < n:
        if y[i] > y[i - 1]:  # rising again → done
            break
        if y[i] < y[valley]:
            valley = i
        i += 1

    if i == n:  # never rose again
        return None

    return peak, valley


def nll(amp, mu, sigma, bin_centers_fit, n_fit):
    """Poisson NLL for binned data"""
    expected = gaussian(bin_centers_fit, amp, mu, sigma)
    expected[expected <= 0] = 1e-10  # avoid log(0)
    return np.sum(expected - n_fit * np.log(expected))


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def linear_func(x, a, b):
    return a * x + b


if __name__ == "__main__":

    f_aux = "/home/pkrause/noise_hunt/data/p-1-1-om-hs-31/ref-v0.0.0/generated/tier/aux/r020/p-1-1-om-hs-31.yaml"
    f_result = "/home/pkrause/software/mint-analysis/debug_out/results.yaml"
    with open(f_aux) as file:
        aux_dict = yaml.safe_load(file)

    raw_path = (
        "/home/pkrause/noise_hunt/data/p-1-1-om-hs-31/ref-v0.0.0/generated/tier/raw/r020/r020"
    )
    plot_folder = "/home/pkrause/software/mint-analysis/debug_out/pe_spectra/"

    override_results = True

    bin_size = 20
    bins = np.arange(-100, 10000, bin_size)
    lim = 20
    A4_LANDSCAPE = (11.69, 8.27)
    result_dic = {}

    for run in aux_dict:
        result_dic[run] = {}
        f_raw = raw_path + aux_dict[run]["daq"].split("/")[-1].replace("daq", "").replace(
            "data", "lh5"
        )
        f_dsp = f_raw.replace("raw", "dsp")

        with PdfPages(plot_folder + f"pe_spectra_{run}.pdf") as pdf:
            for ch in lh5.ls(f_dsp):
                pmt = int(ch[2:]) + 1
                result_dic[run][pmt] = {}
                result_dic[run][pmt]["voltage"] = {
                    "val": aux_dict[run]["voltages_in_V"][int(ch[2:])],
                    "unit": "V",
                }
                fig, ax = plt.subplots()
                fig.set_figwidth(A4_LANDSCAPE[0])
                fig.set_figheight(A4_LANDSCAPE[1])

                d = lh5.read_as(f"{ch}/dsp", f_dsp, "ak")
                vals = ak.to_numpy(d.nnls_solution.values, allow_missing=False)
                pe_vals = np.nansum(np.where(vals > lim, vals, np.nan), axis=1)

                n, bins, patches = ax.hist(
                    pe_vals,
                    bins=bins,
                    histtype="step",
                    label=f"channel {ch} ON ({result_dic[run][pmt]['voltage']['val']:.2f} {result_dic[run][pmt]['voltage']['unit']})",
                )

                if aux_dict[run]["voltages_in_V"][int(ch[2:])] == 0:
                    ax.set_xlim(-10, 2.5e3)
                    ax.set_ylim(0.5, None)
                    ax.set_yscale("log")
                    ax.set_ylabel(f"Counts/{bin_size} NNLS units")
                    ax.legend()
                    ax.set_xlabel("NNLS units")
                    ax.set_title("pygama-NNLS reconstruction (20 units solution cut-off)")

                    pdf.savefig()
                    plt.close()
                    continue

                ######################
                # Fit p.e. peak      #
                ######################

                noise_peak, valley_idx = valley_index_strict(n)
                pe_peak, _ = valley_index_strict(n[noise_peak:])
                ax.axvline(bins[valley_idx], color="red", ls="--")
                ax.axvline(bins[noise_peak:][pe_peak], color="green", ls="--")

                bin_centers = 0.5 * (bins[1:] + bins[:-1])
                # Restrict range
                x_min, x_max = (
                    bins[valley_idx],
                    2 * bins[noise_peak:][pe_peak],
                )  # - bins[valley_idx]
                mask = (bin_centers >= x_min) & (bin_centers <= x_max)

                bin_centers_fit = bin_centers[mask]
                n_fit = n[mask]

                # Initial guesses
                amp0 = n[noise_peak:][pe_peak]
                mu0 = bins[noise_peak:][pe_peak]
                sigma0 = 100.0

                m = Minuit(
                    lambda amp, mu, sigma: nll(
                        amp, mu, sigma, bin_centers_fit=bin_centers_fit, n_fit=n_fit
                    ),
                    amp=amp0,
                    mu=mu0,
                    sigma=sigma0,
                )
                m.errordef = Minuit.LIKELIHOOD  # important for Poisson NLL
                m.migrad(iterate=10)  # perform minimization

                fit_vals = m.values.to_dict()
                fit_errs = m.errors.to_dict()

                result_dic[run][pmt]["pe_peak_fit"] = {
                    "mean": {"val": fit_vals["mu"], "err": fit_errs["mu"], "unit": "NNLS"},
                    "sigma": {"val": fit_vals["sigma"], "err": fit_errs["sigma"], "unit": "NNLS"},
                    "amp": {"val": fit_vals["amp"], "err": fit_errs["amp"], "unit": ""},
                }

                y_fit = gaussian(
                    bin_centers, amp=m.values["amp"], mu=m.values["mu"], sigma=m.values["sigma"]
                )

                ax.plot(bin_centers, y_fit, "r-", label="NLL fit (Minuit)")

                ax.set_xlim(-10, 2.5e3)
                ax.set_ylim(0.5, None)
                ax.set_yscale("log")
                ax.set_ylabel(f"Counts/{bin_size} NNLS units")
                ax.legend()
                ax.set_xlabel("NNLS units")
                ax.set_title("pygama-NNLS reconstruction (20 units solution cut-off)")

                pdf.savefig()
                plt.close()

                ######################
                # spectrum values    #
                ######################

                result_dic[run][pmt]["statistics"] = {
                    "1st_pe_fit_integral_below_valley": {
                        "val": float(np.sum(y_fit[:valley_idx])),
                        "unit": "",
                    },
                    "cts_above_valley": {"val": int(np.sum(n[:valley_idx])), "unit": ""},
                    "cts_below_valley": {"val": int(np.sum(n[valley_idx:])), "unit": ""},
                    "1st_pe_fit_integral": {"val": int(float(np.sum(y_fit))), "unit": ""},
                    "total_cts": {"val": int(np.sum(n)), "unit": ""},
                    "valley": {
                        "pos": {"val": float(bin_centers[valley_idx]), "unit": "NNLS"},
                        "amp": int(n[valley_idx]),
                    },
                    "1st_pe_guess": {
                        "pos": {"val": float(mu0), "unit": "NNLS"},
                        "amp": {"val": int(amp0), "unit": ""},
                    },
                }

                result_dic[run][pmt]["runtime"] = {"unit": "s"}
                if "runtime_in_s" in aux_dict[run]:
                    result_dic[run][pmt]["runtime"]["aux"] = aux_dict[run]["runtime_in_s"]

                if f"{ch}/raw/timestamp_sec" in lh5.ls(
                    f_raw, f"{ch}/raw/"
                ) and f"{ch}/raw/timestamp_ps" in lh5.ls(f_raw, f"{ch}/raw/"):
                    ts = lh5.read_as(f"{ch}/raw/timestamp_sec", f_raw, "np")
                    ts_ps = lh5.read_as(f"{ch}/raw/timestamp_ps", f_raw, "np")
                    result_dic[run][pmt]["runtime"]["raw"] = float(
                        ts[ts.argmax()]
                        + ts_ps[ts.argmax()] * 1e-12
                        - (ts[ts.argmin()] + ts_ps[ts.argmin()] * 1e-12)
                    )

                # support for old raw files
                elif f"{ch}/raw/timestamp" in lh5.ls(f_raw, f"{ch}/raw/"):
                    ts = lh5.read_as(f"{ch}/raw/timestamp", f_raw, "np")
                    result_dic[run][pmt]["runtime"]["raw"] = float((ts.max() - ts.min()) * 4.8e-9)

if os.path.exists(f_result):
    with open(f_result) as file:
        tmp_dic = yaml.safe_load(file)
    if "pe_spectrum" in tmp_dic.keys() and not override_results:
        raise RuntimeError("Already exists and override flag is not set")
    else:
        tmp_dic["pe_spectrum"] = result_dic
        with open(f_result, "w") as file:
            yaml.safe_dump(tmp_dic, file, default_flow_style=False)

else:
    with open(f_result, "w") as file:
        yaml.safe_dump({"pe_spectrum": result_dic}, file, default_flow_style=False)
