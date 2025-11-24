# Imports
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit

# LEGEND specific imports
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


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


if __name__ == "__main__":

    f_raw = "/home/pkrause/noise_hunt/data/p-1-1-om-hs-31/ref-v0.0.0/generated/tier/raw/r020/r020_2025_10_22_23_43_47.lh5"
    f_dsp = f_raw.replace("raw", "dsp")

    bin_size = 20
    bins = np.arange(-100, 10000, bin_size)
    lim = 20
    A4_LANDSCAPE = (11.69, 8.27)
    result_dic = {}

    with PdfPages("pe_spectra.pdf") as pdf:
        for ch in lh5.ls(f_dsp):
            result_dic[ch] = {}
            fig, ax = plt.subplots()
            fig.set_figwidth(A4_LANDSCAPE[0])
            fig.set_figheight(A4_LANDSCAPE[1])

            d = lh5.read_as(f"{ch}/dsp", f_dsp, "ak")
            vals = ak.to_numpy(d.nnls_solution.values, allow_missing=False)
            pe_vals = np.nansum(np.where(vals > lim, vals, np.nan), axis=1)

            n, bins, patches = ax.hist(
                pe_vals, bins=bins, histtype="step", label=f"channel {ch} ON ($V_N$)"
            )

            noise_peak, valley_idx = valley_index_strict(n)
            pe_peak, _ = valley_index_strict(n[noise_peak:])
            ax.axvline(bins[valley_idx], color="red", ls="--")
            ax.axvline(bins[noise_peak:][pe_peak], color="green", ls="--")

            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            # Restrict range
            x_min, x_max = bins[valley_idx], 2 * bins[noise_peak:][pe_peak]  # - bins[valley_idx]
            mask = (bin_centers >= x_min) & (bin_centers <= x_max)

            bin_centers_fit = bin_centers[mask]
            n_fit = n[mask]

            def nll(amp, mu, sigma):
                """Poisson NLL for binned data"""
                expected = gaussian(bin_centers_fit, amp, mu, sigma)
                expected[expected <= 0] = 1e-10  # avoid log(0)
                return np.sum(expected - n_fit * np.log(expected))

            # Initial guesses
            amp0 = n[noise_peak:][pe_peak]
            mu0 = bins[noise_peak:][pe_peak]
            sigma0 = 100.0

            m = Minuit(nll, amp=amp0, mu=mu0, sigma=sigma0)
            m.errordef = Minuit.LIKELIHOOD  # important for Poisson NLL
            m.migrad(iterate=10)  # perform minimization

            fit_vals = m.values.to_dict()
            fit_errs = m.errors.to_dict()

            result_dic[ch]["pe_peak"] = {"vals": fit_vals, "errs": fit_errs}

            y_fit = gaussian(
                bin_centers, amp=m.values["amp"], mu=m.values["mu"], sigma=m.values["sigma"]
            )

            ax.plot(bin_centers, y_fit, "r-", label="NLL fit (Minuit)")

            dcts = np.sum(y_fit[:valley_idx]) + np.sum(n[valley_idx:])

            # timestamps in rawfile are currently broken
            # ts = lh5.read_as(f"{ch}/raw/timestamp",f_raw,"np")
            dt = 25.021  # from log file

            result_dic[ch]["dcr"] = {"dcr": {"dcr": dcts / dt, "counts": dcts, "runtime_s": dt}}

            ax.set_xlim(-10, 2.5e3)
            ax.set_ylim(0.5, None)
            ax.set_yscale("log")
            ax.set_ylabel(f"Counts/{bin_size} NNLS units")
            ax.legend()
            ax.set_xlabel("NNLS units")
            ax.set_title("pygama-NNLS reconstruction (20 units solution cut-off)")

            pdf.savefig()
            plt.close()

    print(result_dic)
