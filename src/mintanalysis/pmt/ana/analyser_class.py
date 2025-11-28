"""
PESpectrumAnalyzer

- Logging to stdout + file.
- Missing critical files -> hard fail.
- Per-channel fit/peak failures -> skip and continue.
- One PDF per run (written once; pages appended in-memory during run processing).
- Cleaner decomposition and minimal repetition.
- 1st p.e. fit: single gauss
- DCR estimate and plot
- linear gain fit
- SNR plot

Usage: Run via CLI with desired flags
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import yaml
from iminuit import Minuit
from lgdo import lh5
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# --------------------------
# TODO For Debugging only!
# --------------------------
RAW_DIR = Path("/home/pkrause/noise_hunt/data/p-1-1-om-hs-31/ref-v0.0.0/generated/tier/raw/r020")
RESULT_DIR = Path("/home/pkrause/software/mint-analysis/debug_out")

# --------------------------
# Constants
# --------------------------
A4_LANDSCAPE = (11.69, 8.27)


# --------------------------
# Logging Setup
# --------------------------


def setup_logging(
    log_file: Path = RESULT_DIR / "analysis.log", level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger("PESpectrum")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] [%(name)s - %(funcName)s] [%(levelname)s] %(message)s"
        )

        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# --------------------------
# Small helpers
# --------------------------


@dataclass
class ChannelResult:
    status: str  # 'ok', 'skipped', 'error'
    data: dict[str, Any]


def linear_func(x, a, b):
    return a * x + b


def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def poisson_nll(amp: float, mu: float, sigma: float, x: np.ndarray, y: np.ndarray) -> float:
    expected = gaussian(x, amp, mu, sigma)
    expected = np.clip(expected, 1e-10, None)
    return float(np.sum(expected - y * np.log(expected)))


def valley_index_strict(y: np.ndarray) -> tuple[int, int] | None:
    """Return (peak_idx, valley_idx) or None."""
    n = len(y)
    if n < 3:
        return None

    peak = None
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peak = i
            break
    if peak is None:
        return None

    valley = peak
    i = peak + 1
    while i < n:
        if y[i] > y[i - 1]:
            return peak, valley
        if y[i] < y[valley]:
            valley = i
        i += 1

    return None


# --------------------------
# Analyzer class
# --------------------------
class PESpectrumAnalyzer:
    def __init__(
        self,
        aux_yaml: Path,
        bin_size: int = 20,
        lim: float = 20,
        override_results: bool = False,
        logger: logging.Logger | None = None,
        up_sampling_ratio: float = 24 / 240,
        v_per_adc: float = 0.25e-3,
        adc_impedance: float = 75.0,
        sampling_time: float = 4.8e-9,
        calib: str = "None",
    ) -> None:
        self.aux_yaml = aux_yaml
        self.raw_dir = RAW_DIR
        self.plot_folder = RESULT_DIR / "plots"
        self.result_yaml = RESULT_DIR / "results.yaml"
        self.bin_size = bin_size
        self.bins = np.arange(-100, 10000, bin_size)
        self.lim = lim
        self.override_results = override_results
        self.logger = logger or setup_logging()
        self.aux = self._load_aux()
        self.plot_folder.mkdir(parents=True, exist_ok=True)
        self.up_sampling_ratio = up_sampling_ratio
        self.v_per_adc = v_per_adc
        self.adc_impedance = adc_impedance
        self.sampling_time = sampling_time
        self.calib = calib

    # ----------------------
    # I/O helpers
    # ----------------------
    def _load_aux(self) -> dict[str, Any]:
        if not self.aux_yaml.exists():
            msg = f"Aux file not found: {self.aux_yaml}"
            raise FileNotFoundError(msg)
        with open(self.aux_yaml) as f:
            aux = yaml.safe_load(f)
        self.logger.info("Loaded aux YAML: %s", self.aux_yaml)
        return aux

    # ----------------------
    # Public entrypoints
    # ----------------------
    def run(self) -> None:
        results: dict[str, dict[int, dict[str, Any]]] = {}
        for run_name, meta in self.aux.items():
            self.logger.info("Starting run: %s", run_name)
            try:
                results[run_name] = self.analyze_run(run_name, meta)
            except FileNotFoundError as e:
                self.logger.exception("Critical file missing for run %s: %s", run_name, e)
                raise
        self._save_results(results)
        self.logger.info("All runs processed.")

    # ----------------------
    # Per-run flow
    # ----------------------
    def analyze_run(self, run_name: str, meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
        # build file paths
        fname = meta["daq"].split("/")[-1].replace("daq", "r020").replace("data", "lh5")
        f_raw = self.raw_dir / fname
        if not f_raw.exists():
            msg = f"Raw file for run {run_name} not found: {f_raw}"
            raise FileNotFoundError(msg)
        f_dsp = Path(str(f_raw).replace("raw", "dsp"))
        if not f_dsp.exists():
            msg = f"DSP file for run {run_name} not found: {f_dsp}"
            raise FileNotFoundError(msg)

        run_results: dict[int, dict[str, Any]] = {}
        pdf_path = self.plot_folder / f"pe_spectra_{run_name}.pdf"

        # collect figures and write once
        with PdfPages(pdf_path) as pdf:
            for ch in lh5.ls(f_dsp):
                pmt = int(ch[2:]) + 1
                self.logger.info("Run %s - channel %s (PMT %d)", run_name, ch, pmt)
                try:
                    fig, chan_data = self.process_channel(run_name, ch, pmt, meta, f_raw, f_dsp)
                    # fig may be None if plotting skipped
                    if fig is not None:
                        pdf.savefig(fig)
                        plt.close(fig)
                    run_results[pmt] = chan_data
                except Exception as exc:
                    self.logger.exception(
                        "Channel-level error run=%s ch=%s: %s", run_name, ch, exc
                    )
                    run_results[pmt] = {"status": "error", "reason": str(exc)}

        self.logger.info("Wrote PDF for run %s to %s", run_name, pdf_path)
        return run_results

    # ----------------------
    # Per-channel processing
    # ----------------------
    def process_channel(
        self,
        run_name: str,
        ch: str,
        pmt: int,
        meta: dict[str, Any],
        f_raw: Path,
        f_dsp: Path,
    ) -> tuple[plt.Figure | None, dict[str, Any]]:
        """Process channel. Returns (figure_or_None, channel_result_dict).

        Non-critical failures return a result dict with status 'skipped'
        """
        result: dict[str, Any] = {}
        ch_idx = int(ch[2:])
        voltage = float(meta["voltages_in_V"][ch_idx])
        result["voltage"] = {"val": voltage, "unit": "V"}

        # load data
        try:
            d = lh5.read_as(f"{ch}/dsp", f_dsp, "ak")
        except Exception as e:
            msg = f"Failed to read DSP for {ch} in {f_dsp}: {e}"
            self.logger.warning(msg)
            return None, {"status": "skipped", "reason": msg}

        # compute pe-values
        try:
            vals = ak.to_numpy(d.nnls_solution.values, allow_missing=False)
            pe_vals = np.nansum(np.where(vals > self.lim, vals, np.nan), axis=1)
        except Exception as e:
            msg = f"Failed to compute pe values for {ch}: {e}"
            self.logger.warning(msg)
            return None, {"status": "skipped", "reason": msg}

        # histogram
        fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
        n, bins, _ = ax.hist(
            pe_vals,
            bins=self.bins,
            histtype="step",
            label=f"channel {ch} (PMT {pmt}) at {voltage:.2f} V",
        )

        if self.calib != "None":
            self._add_charge_axis(ax, False)

        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # noise-only
        if voltage == 0:
            self._decorate_axis(ax)
            # minimal result (no fit)
            result.update(
                {"status": "ok", "statistics": {}, "pe_peak_fit": {}, "runtime": {"unit": "s"}}
            )
            if "runtime_in_s" in meta:
                result["runtime"]["aux"] = meta["runtime_in_s"]
            raw_runtime = self._extract_runtime_if_present(f_raw, ch)
            if raw_runtime is not None:
                result["runtime"]["raw"] = raw_runtime
            return fig, result

        # detect valley & peaks
        vi = valley_index_strict(n)
        if vi is None:
            msg = "Valley detection failed (no strict peak/valley)."
            self.logger.warning("Run %s ch %s: %s", run_name, ch, msg)
            self._decorate_axis(ax)
            return fig, {"status": "skipped", "reason": msg}

        noise_peak, valley_idx = vi

        # find first p.e. peak after noise_peak
        sub = n[noise_peak:]
        pe_vi = valley_index_strict(sub)
        if pe_vi is None:
            msg = "1st-p.e. detection failed after noise peak."
            self.logger.warning("Run %s ch %s: %s", run_name, ch, msg)
            self._decorate_axis(ax)
            return fig, {"status": "skipped", "reason": msg}

        pe_peak_rel, _ = pe_vi
        pe_peak_idx = noise_peak + pe_peak_rel

        # annotate choices
        ax.axvline(bin_centers[valley_idx], color="red", ls="--", label="valley")
        ax.axvline(bin_centers[pe_peak_idx], color="green", ls="--", label="1st pe guess")

        # fit window
        x_min = bin_centers[valley_idx]
        x_max = 2 * bin_centers[pe_peak_idx]
        mask = (bin_centers >= x_min) & (bin_centers <= x_max)
        bin_centers_fit = bin_centers[mask]
        n_fit = n[mask]

        if len(bin_centers_fit) < 3:
            msg = f"Insufficient bins for fitting (n_fit={len(bin_centers_fit)})."
            self.logger.warning("Run %s ch %s: %s", run_name, ch, msg)
            self._decorate_axis(ax)
            return fig, {"status": "skipped", "reason": msg}

        amp0 = float(n[pe_peak_idx])
        mu0 = float(bin_centers[pe_peak_idx])
        sigma0 = 100.0

        try:
            m = Minuit(
                lambda amp, mu, sigma: poisson_nll(amp, mu, sigma, bin_centers_fit, n_fit),
                amp=amp0,
                mu=mu0,
                sigma=sigma0,
            )
            m.errordef = Minuit.LIKELIHOOD
            m.migrad(iterate=10)
        except Exception as e:
            msg = f"Minuit error for {ch}: {e}"
            self.logger.warning("Run %s ch %s: %s", run_name, ch, msg)
            self._decorate_axis(ax)
            return fig, {"status": "skipped", "reason": msg}

        # Basic validity check
        if not getattr(m, "valid", True):
            self.logger.warning("Minuit invalid result for run %s ch %s", run_name, ch)
            self._decorate_axis(ax)
            return fig, {"status": "skipped", "reason": "Minuit invalid"}

        fit_vals = {k: float(v) for k, v in m.values.to_dict().items()}
        fit_errs = {k: float(v) for k, v in m.errors.to_dict().items()}

        result["pe_peak_fit"] = {
            "mean": {"val": fit_vals["mu"], "err": fit_errs["mu"], "unit": "NNLS"},
            "sigma": {"val": fit_vals["sigma"], "err": fit_errs["sigma"], "unit": "NNLS"},
            "amp": {"val": fit_vals["amp"], "err": fit_errs["amp"], "unit": ""},
        }

        # full fit curve over bins for plotting & integrals
        y_fit = gaussian(bin_centers, fit_vals["amp"], fit_vals["mu"], fit_vals["sigma"])
        ax.plot(bin_centers, y_fit, "r-", label="NLL fit (Minuit)")

        self._decorate_axis(ax)

        # statistics
        try:
            result["statistics"] = {
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
                    "amp": {"val": int(n[valley_idx]), "unit": ""},
                },
                "1st_pe_guess": {
                    "pos": {"val": float(mu0), "unit": "NNLS"},
                    "amp": {"val": int(amp0), "unit": ""},
                },
            }
        except Exception as e:
            self.logger.warning(
                "Statistics computation failed for run %s ch %s: %s", run_name, ch, e
            )
            result.setdefault("statistics", {})
            result["statistics"]["error"] = str(e)

        # runtime
        result["runtime"] = {"unit": "s"}
        if "runtime_in_s" in meta:
            result["runtime"]["aux"] = meta["runtime_in_s"]
        raw_runtime = self._extract_runtime_if_present(f_raw, ch)
        if raw_runtime is not None:
            result["runtime"]["raw"] = raw_runtime

        result["status"] = "ok"
        return fig, result

    # ----------------------
    # Utilities
    # ----------------------
    def _decorate_axis(self, ax: plt.Axes) -> None:
        ax.set_xlim(-10, 2.5e3)
        ax.set_ylim(0.5, None)
        ax.set_yscale("log")
        ax.set_ylabel(f"Counts/{self.bin_size} NNLS units")
        ax.set_xlabel("NNLS units")
        ax.set_title(f"pygama-NNLS reconstruction ({self.lim} units solution cut-off)")
        ax.legend()

    def _extract_runtime_if_present(self, f_raw: Path, ch: str) -> float | None:
        try:
            entries = set(lh5.ls(f_raw, f"{ch}/raw/"))
        except Exception:
            entries = set()

        sec_key = f"{ch}/raw/timestamp_sec"
        ps_key = f"{ch}/raw/timestamp_ps"
        if sec_key in entries and ps_key in entries:
            try:
                ts = lh5.read_as(sec_key, f_raw, "np")
                ts_ps = lh5.read_as(ps_key, f_raw, "np")
                idx_max = int(ts.argmax())
                idx_min = int(ts.argmin())
                return float(
                    ts[idx_max] + ts_ps[idx_max] * 1e-12 - (ts[idx_min] + ts_ps[idx_min] * 1e-12)
                )
            except Exception:
                self.logger.debug("Failed to extract new-style timestamps for %s in %s", ch, f_raw)

        legacy_key = f"{ch}/raw/timestamp"
        if legacy_key in entries:
            try:
                t = lh5.read_as(legacy_key, f_raw, "np")
                return float((t.max() - t.min()) * 4.8e-9)
            except Exception:
                self.logger.debug("Failed to extract legacy timestamps for %s in %s", ch, f_raw)

        return None

    def _save_results(self, results: dict[str, dict[int, dict[str, Any]]]) -> None:
        if self.result_yaml.exists():
            with open(self.result_yaml) as f:
                existing = yaml.safe_load(f) or {}
            if "pe_spectrum" in existing and not self.override_results:
                msg = "Results already present and override flag is False."
                raise RuntimeError(msg)
            existing["pe_spectrum"] = results
            with open(self.result_yaml, "w") as f:
                yaml.safe_dump(existing, f, default_flow_style=False)
            self.logger.info("Updated result YAML at %s", self.result_yaml)
        else:
            with open(self.result_yaml, "w") as f:
                yaml.safe_dump({"pe_spectrum": results}, f, default_flow_style=False)
            self.logger.info("Wrote new result YAML at %s", self.result_yaml)

    def _add_charge_axis(self, ax: plt.Axes, is_y) -> None:
        """
        This function add a axis in new units to a given axis plot.

        Parameters
        ----------
        ax : plt.Axes
            The plot to which to add the pC axis.
        is_y : bool
            If True add a new y-axis. Else an x-axis.
        """
        if self.calib not in ["pC", "adc", "gain"]:
            self.logger.warning("Invalid calibration unit, not calibrating.")
            return

        label = "Charge "
        if self.calib == "pC":
            func = (
                lambda x: x
                * (
                    (self.v_per_adc * self.up_sampling_ratio * self.sampling_time)
                    / self.adc_impedance
                )
                * 1e12,
                lambda y: y
                / (
                    (
                        (self.v_per_adc * self.up_sampling_ratio * self.sampling_time)
                        / self.adc_impedance
                    )
                    * 1e12
                ),
            )
            label += "(pC)"

        elif self.calib == "gain":
            label = "Gain (a.u.)"
            elem = 1.602e-19  # C (elementary charge)
            func = (
                lambda x: x
                * (
                    (self.v_per_adc * self.up_sampling_ratio * self.sampling_time)
                    / self.adc_impedance
                )
                / elem,
                lambda y: y
                / (
                    (
                        (self.v_per_adc * self.up_sampling_ratio * self.sampling_time)
                        / self.adc_impedance
                    )
                    / elem
                ),
            )

        elif self.calib == "adc":
            label += f"(ADC x {self.sampling_time*1e9:.1f} ns)"
            func = (lambda x: x * self.up_sampling_ratio, lambda y: y / self.up_sampling_ratio)
        if is_y:
            secax_y = ax.secondary_yaxis("right", functions=func)
            secax_y.set_ylabel(label)
        else:
            secax_x = ax.secondary_xaxis("top", functions=func)
            secax_x.set_xlabel(label)

    # ----------------------
    # Signal to Noise Ratio (SNR)
    # ----------------------
    def plot_snr(self) -> None:
        if not self.result_yaml.exists():
            msg = f"Result YAML not found: {self.result_yaml}"
            raise FileNotFoundError(msg)
        with open(self.result_yaml) as f:
            data = yaml.safe_load(f) or {}
        if "pe_spectrum" not in data:
            msg = "pe_spectrum info not present in result YAML; run analysis first."
            raise RuntimeError(msg)

        snr = {}
        for run_name, pmt_dict in data["pe_spectrum"].items():
            run_snr = {}
            for pmt, info in pmt_dict.items():
                try:
                    if info.get("voltage", {}).get("val", 0) == 0:
                        continue
                    noise = {
                        "val": info.get("statistics", {})
                        .get("valley", {})
                        .get("amp", {})
                        .get("val", 0.0)
                    }
                    noise["err"] = noise["val"] ** 0.5
                    if info.get("status", "skipped") == "ok":
                        signal = {
                            "val": info.get("pe_peak_fit").get("amp", {}).get("val", 0.0),
                            "err": info.get("pe_peak_fit").get("amp", {}).get("err", 0.0),
                        }
                    else:
                        signal = {
                            "val": info.get("statistics", {})
                            .get("1st_pe_guess", {})
                            .get("amp", {})
                            .get("val", 0)
                        }
                        signal["err"] = signal["val"] ** 0.5
                    run_snr[pmt] = {
                        "val": 1 - noise["val"] / signal["val"],
                        "err": (
                            noise["err"] ** 2 / signal["val"] ** 2
                            + (noise["val"] ** 2 * signal["err"] ** 2) / signal["val"] ** 4
                        ),
                        "unit": "",
                    }
                except Exception as e:
                    self.logger.warning(
                        "Failed to compute SNR for run %s PMT %s: %s", run_name, pmt, e
                    )
            snr[run_name] = run_snr

        fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
        for run_name, pmt_dict in snr.items():
            pmts = sorted(pmt_dict.keys())
            vals = [pmt_dict[p]["val"] for p in pmts]
            ax.plot(pmts, vals, marker="o", label=run_name)
        ax.set_xlabel("PMT")
        ax.set_ylabel("SNR (a.u.)")
        ax.set_title("SNR per PMT (= 1 - valley/peak)")
        ax.legend()
        plt.tight_layout()
        plot_path = self.plot_folder / "snr_plot.png"
        fig.savefig(plot_path)
        plt.close(fig)
        self.logger.info("SNR plot saved to %s", plot_path)

    # ----------------------
    # Dark Count Rate (DCR)
    # ----------------------
    def compute_dark_count_rate(self, time_mode: str = "aux") -> None:
        if not self.result_yaml.exists():
            msg = f"Result YAML not found: {self.result_yaml}"
            raise FileNotFoundError(msg)
        with open(self.result_yaml) as f:
            data = yaml.safe_load(f) or {}
        if "pe_spectrum" not in data:
            msg = "pe_spectrum info not present in result YAML; run analysis first."
            raise RuntimeError(msg)
        if "dcr" in data and not self.override_results:
            msg = "DCR already exists and override flag is False."
            raise RuntimeError(msg)
        dcr = {}
        for run_name, pmt_dict in data["pe_spectrum"].items():
            run_dcr = {}
            for pmt, info in pmt_dict.items():
                try:
                    if info.get("voltage", {}).get("val", 0) == 0:
                        continue
                    stats = info.get("statistics", {})
                    runtime_info = info.get("runtime", {})
                    if time_mode not in runtime_info:
                        self.logger.warning(
                            "Run %s PMT %s: missing runtime '%s'; skipping DCR.",
                            run_name,
                            pmt,
                            time_mode,
                        )
                        continue
                    dcts = stats.get("1st_pe_fit_integral_below_valley", {}).get(
                        "val", 0.0
                    ) + stats.get("cts_above_valley", {}).get("val", 0)
                    runtime = runtime_info[time_mode]
                    run_dcr[pmt] = {
                        "val": float(dcts) / float(runtime),
                        "err": float(dcts**0.5) / float(runtime),
                        "unit": "Hz",
                    }
                except Exception as e:
                    self.logger.warning(
                        "Failed to compute DCR for run %s PMT %s: %s", run_name, pmt, e
                    )
            dcr[run_name] = run_dcr
        data["dcr"] = dcr
        with open(self.result_yaml, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
        self.logger.info("Wrote DCR results to %s", self.result_yaml)

        fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
        for run_name, pmt_dict in dcr.items():
            pmts = sorted(pmt_dict.keys())
            vals = [pmt_dict[p]["val"] for p in pmts]
            ax.plot(pmts, vals, marker="o", label=run_name)
        ax.set_xlabel("PMT")
        ax.set_ylabel("DCR (Hz)")
        ax.set_title("Dark Count Rate per PMT")
        ax.legend()
        plt.tight_layout()
        plot_path = self.plot_folder / "dcr_plot.png"
        fig.savefig(plot_path)
        plt.close(fig)
        self.logger.info("DCR plot saved to %s", plot_path)

    # ----------------------
    # Linear Gain Fit computation
    # ----------------------
    def compute_linear_gain_fit(self) -> None:
        if not self.result_yaml.exists():
            msg = f"Result YAML not found: {self.result_yaml}"
            raise FileNotFoundError(msg)
        with open(self.result_yaml) as f:
            data = yaml.safe_load(f) or {}
        if "pe_spectrum" not in data:
            msg = "pe_spectrum info not present in result YAML; run analysis first."
            raise RuntimeError(msg)

        tmp_dic = {"used_keys": []}
        y_unit = None
        x_unit = None
        for key, run in data["pe_spectrum"].items():
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

        pdf_path = self.plot_folder / "gain_plots.pdf"
        with PdfPages(pdf_path) as pdf:
            for key, pmt in tmp_dic.items():
                if key == "used_keys":
                    continue
                fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
                ax.errorbar(pmt["voltage"], pmt["vals"], pmt["errs"], fmt="o", label=f"PMT {key}")
                ax.set_xlabel(f"Voltage ({x_unit})")
                ax.set_ylabel(f"PMT position ({y_unit})")

                if self.calib != "None":
                    self._add_charge_axis(ax, True)

                params, covariance = curve_fit(
                    linear_func,
                    pmt["voltage"],
                    pmt["vals"],
                    sigma=pmt["errs"],
                    absolute_sigma=True,
                )
                a_opt, b_opt = params
                perr = np.sqrt(np.diag(covariance))
                x = np.linspace(-1 * b_opt / a_opt, max(pmt["voltage"]) + 10, 1000)
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
                pmt.pop("voltage")
                pmt.pop("vals")
                pmt.pop("errs")
                ax.legend()
                pdf.savefig(fig)
                plt.close(fig)
        data["linear_gain"] = tmp_dic
        with open(self.result_yaml, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
        self.logger.info("Linear gain fit results saved to %s", self.result_yaml)

    def calibrate_nnls_values(self, calibration_func, output_file, new_unit):
        """
        Reads the current results.yaml, finds all entries that contain a dict with
        keys {"val": <number>, "unit": "NNLS"}, applies calibration_func(val)
        and writes a calibrated result file.
        """
        factor = calibration_func(1)
        try:
            with open(self.result_yaml) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            msg = f"Failed to load file: {e}"
            self.logger.error(msg)
            raise

        def classify(unit: str):
            unit = str(unit)
            if unit == "NNLS" or unit.startswith("NNLS/"):
                return "nnls"
            if "/NNLS" in unit:
                return "per_nnls"
            if "NNLS" in unit:
                return "nnls"
            return None

        def recurse(obj):
            if isinstance(obj, dict):
                if set(obj.keys()) >= {"val", "unit"}:
                    kind = classify(obj.get("unit"))
                    if kind == "nnls":
                        try:
                            obj["val"] = obj["val"] * factor
                            if "err" in obj:
                                obj["err"] = obj["err"] * factor
                            obj["unit"] = obj["unit"].replace("NNLS", new_unit)
                        except Exception as e:
                            msg = f"NNLS calibration failed for value {obj}: {e}"
                            self.logger.error(msg)
                    elif kind == "per_nnls":
                        try:
                            obj["val"] = obj["val"] / factor
                            if "err" in obj:
                                obj["err"] = obj["err"] / factor
                            obj["unit"] = obj["unit"].replace("NNLS", new_unit)
                        except Exception as e:
                            msg = f"per-NNLS calibration failed for value {obj}: {e}"
                            self.logger.error(msg)
                for _, v in obj.items():
                    recurse(v)
            elif isinstance(obj, list):
                for item in obj:
                    recurse(item)

        recurse(data)

        try:
            with open(output_file, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
        except Exception as e:
            msg = f"Failed to write calibrated file: {e}"
            self.logger.error(msg)
            raise

        msg = f"Calibrated results written to {output_file}"
        self.logger.info(msg)


# --------------------------
# CLI entrypoint
# --------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PE Spectrum Analyzer with DCR and linear gain fit"
    )
    parser.add_argument(
        "-p", "--compute-pe", action="store_true", help="Do p.e. spectrum analysis"
    )
    parser.add_argument(
        "-d", "--compute-dcr", action="store_true", help="Compute DCR after analysis"
    )
    parser.add_argument(
        "-g", "--compute-gain", action="store_true", help="Compute linear gain fit after analysis"
    )
    parser.add_argument(
        "-s", "--compute-snr", action="store_true", help="Compute SNR after analysis"
    )
    parser.add_argument(
        "-c",
        "--calibrate",
        default="None",
        choices=["pC", "adc", "gain", "None"],
        help="Choose a charge calibration value (pC, adc, gain, or None)",
    )
    parser.add_argument(
        "-b",
        "--bin_size",
        type=int,
        default=20,
        help="Number of bins used for analysis",
    )
    parser.add_argument(
        "-l",
        "--nnls_limit",
        type=float,
        default=20,
        help="Lower limit for solutions in the NNLS solution vector to be accepted.",
    )
    parser.add_argument(
        "-a",
        "--aux_file",
        help="Path to auxiliary file",
    )
    parser.add_argument(
        "-o", "--override", action="store_true", help="Override results if existing"
    )

    args = parser.parse_args()

    logger = setup_logging(level=logging.INFO)
    try:
        analyzer = PESpectrumAnalyzer(
            logger=logger,
            aux_yaml=Path(args.aux_file),
            bin_size=args.bin_size,
            lim=args.nnls_limit,
            override_results=args.override,
            calib=args.calibrate,
        )
        if args.compute_pe:
            analyzer.run()
        if args.compute_dcr:
            analyzer.compute_dark_count_rate(time_mode="aux")
        if args.compute_gain:
            analyzer.compute_linear_gain_fit()
        if args.compute_snr:
            analyzer.plot_snr()
        if args.calibrate != "None":
            if args.calibrate == "pC":
                analyzer.calibrate_nnls_values(
                    lambda x: x
                    * (
                        (analyzer.v_per_adc * analyzer.up_sampling_ratio * analyzer.sampling_time)
                        / analyzer.adc_impedance
                    )
                    * 1e12,
                    str(analyzer.result_yaml).replace(".yaml", "_pC_calibrated.yaml"),
                    "pC",
                )
            elif args.calibrate == "adc":
                analyzer.calibrate_nnls_values(
                    lambda x: x * analyzer.up_sampling_ratio,
                    str(analyzer.result_yaml).replace(".yaml", "_adc_calibrated.yaml"),
                    "ADC",
                )
            elif args.calibrate == "gain":
                elem = 1.602e-19  # C (elementary charge)
                analyzer.calibrate_nnls_values(
                    lambda x: x
                    * (
                        (analyzer.v_per_adc * analyzer.up_sampling_ratio * analyzer.sampling_time)
                        / analyzer.adc_impedance
                    )
                    / elem,
                    str(analyzer.result_yaml).replace(".yaml", "_gain_calibrated.yaml"),
                    "a.u.",
                )

        logger.info("Processing complete.")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        raise


# --------------------------
# CLI entrypoint
# --------------------------
if __name__ == "__main__":
    main()
