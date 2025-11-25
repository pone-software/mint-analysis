import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import yaml
from iminuit import Minuit
from lgdo import lh5
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit


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

    with open(
        "/home/pkrause/noise_hunt/data/p-1-1-om-hs-31/ref-v0.0.0/generated/tier/aux/r020/p-1-1-om-hs-31.yaml",
    ) as file:
        aux_dict = yaml.safe_load(file)

    raw_path = (
        "/home/pkrause/noise_hunt/data/p-1-1-om-hs-31/ref-v0.0.0/generated/tier/raw/r020/r020"
    )
    out_folder = "/home/pkrause/software/mint-analysis/debug_out/"

    f_result = "result.yaml"
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

        with PdfPages(out_folder + f"pe_spectra_{run}.pdf") as pdf:
            for ch in lh5.ls(f_dsp):
                pmt = int(ch[2:]) + 1
                result_dic[run][pmt] = {}
                result_dic[run][pmt]["voltages_in_V"] = aux_dict[run]["voltages_in_V"][int(ch[2:])]
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
                    label=f"channel {ch} ON ({result_dic[run][pmt]['voltages_in_V']:.2f} V)",
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

                result_dic[run][pmt]["pe_peak"] = {"vals": fit_vals, "errs": fit_errs}

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
                # DCR calculation    #
                ######################

                dcts = np.sum(y_fit[:valley_idx]) + np.sum(n[valley_idx:])

                # TODO fix timestamps in raw (they are sample numbers currently)
                ts = lh5.read_as(f"{ch}/raw/timestamp", f_raw, "np")
                dt = (ts.max() - ts.min()) * 4.8e-9

                result_dic[run][pmt]["dcr"] = {
                    "dcr": {"dcr": float(dcts / dt), "counts": float(dcts), "runtime_s": float(dt)}
                }

    ######################
    # Gain calculation   #
    ######################
    tmp_dic = {}
    for _, run in result_dic.items():
        for pmt in run:
            if pmt not in tmp_dic:
                tmp_dic[pmt] = {"voltage": [], "vals": [], "errs": []}
            v = run[pmt]["voltages_in_V"]
            if v == 0:
                continue
            tmp_dic[pmt]["voltage"].append(v)
            tmp_dic[pmt]["vals"].append(run[pmt]["pe_peak"]["vals"]["mu"])
            tmp_dic[pmt]["errs"].append(run[pmt]["pe_peak"]["errs"]["mu"])

    with PdfPages(out_folder + "gain_plots.pdf") as pdf:
        for key, pmt in tmp_dic.items():
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
            ax.set_ylabel("PMT position (NNLS units)")
            ax.set_xlabel("Voltage (V)")

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

            tmp_dic[key]["fit"] = {
                "vals": {"a": float(a_opt), "b": float(b_opt)},
                "errs": {"a": float(perr[0]), "b": float(perr[1])},
                "func": "gain_in_NNLS_units = a*voltage_in_V+b",
            }

            ax.legend()
            pdf.savefig()
            plt.close()

    result_dic["gain"] = tmp_dic

with open(out_folder + "results.yaml", "w") as file:
    aux_dict = yaml.safe_dump(result_dic, file, default_flow_style=False)
