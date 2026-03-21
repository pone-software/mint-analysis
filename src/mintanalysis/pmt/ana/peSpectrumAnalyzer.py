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

IceTraySpectrumAnalyzer

Child of PESpectrumAnalyzer to do analysis based on an icecube file
Needs icetray installed

Usage: Run via CLI with desired flags
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from iminuit import Minuit
from lgdo import lh5
from matplotlib.backends.backend_pdf import PdfPages
from pint import UnitRegistry
from scipy.optimize import curve_fit
from uncertainties import ufloat

from .utils import (
    gaussian,
    get_physics_object,
    linear_func,
    poisson_nll,
    quantity_to_dict,
    setup_logging,
    valley_index_strict,
)

# --------------------------
# Constants
# --------------------------
A4_LANDSCAPE = (11.69, 8.27)
CONSIDERED_OFF_VOLTAGE = 12  # volts


@dataclass(frozen=True)
class Calibration:
    up_sampling_ratio: float
    v_per_adc: float
    adc_impedance: float
    sampling_time: float
    renormalization_factor: float


@dataclass
class ChannelResult:
    status: str  # 'ok', 'skipped', 'error'
    data: dict[str, Any]


# --------------------------
# Analyzer class
# --------------------------
class PESpectrumAnalyzer:
    def __init__(
        self,
        aux_yaml: Path,
        keys: list | None = None,
        ignore_keys: list | None = None,
        bin_size: int = 20,
        lim: float = 20,
        override_results: bool = False,
        logger: logging.Logger | None = None,
        calibrator: Calibration | None = None,
        calib: str = "None",
    ) -> None:
        self.aux_yaml = aux_yaml
        self.keys = keys
        self.ignore_keys = ignore_keys
        self._set_up_paths()
        self.bin_size = bin_size
        self.bins = np.arange(-100, 10000, bin_size)
        self.lim = lim
        self.override_results = override_results
        self.hemispheres = {}
        self.logger = logger or setup_logging()

        self.calibrator = calibrator
        if calibrator is None:
            self.calibrator = Calibration(24 / 240, 0.25e-3, 75.0, 4.8e-9, 1)
        self.calib = calib
        self.ureg = UnitRegistry()
        self.aux = self._load_aux()

        # unit handling
        vadc = self.ureg.Quantity(self.calibrator.v_per_adc, self.ureg.volt)
        usr = self.ureg.Quantity(self.calibrator.up_sampling_ratio, self.ureg.dimensionless)
        rf = self.ureg.Quantity(self.calibrator.renormalization_factor, self.ureg.dimensionless)
        st = self.ureg.Quantity(self.calibrator.sampling_time, self.ureg.seconds)
        imp = self.ureg.Quantity(self.calibrator.adc_impedance, self.ureg.ohm)

        cal_dict = {
            "vadc": vadc,
            "upsampling_ratio": usr,
            "renormalization_factor": rf,
            "sampling_time": st,
            "adc_impedance": imp,
        }

        nnls_coloumb_factor = (vadc * usr * st * rf) / imp
        self.ureg.define(f"NNLS = {nnls_coloumb_factor.to('coulomb').magnitude} * coulomb")
        self.ureg.define(f"ADC = {usr.magnitude}*NNLS")

        if self.calib != "None":
            self._save_results(cal_dict, "calibration_constants")

        self._save_results(self.hemispheres, "hemispheres")

    def _set_up_paths(self):
        self.plot_folder = self.aux_yaml.parent / "../ana/plots"
        self.result_yaml = self.aux_yaml.parent / "../ana/results.yaml"
        self.plot_folder.mkdir(parents=True, exist_ok=True)

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

        # convert to physics units
        aux = get_physics_object(aux, self.ureg)
        self.hemispheres = {"A": aux.pop("hemisphere_a"), "B": aux.pop("hemisphere_b")}

        if self.keys is None:
            return aux
        ret = {}
        for k in self.keys:
            if k in aux:
                ret[k] = aux[k]
            else:
                msg = f"Key {k} not in aux file, skipping."
                self.logger.warning(msg)

        for k in ret:
            if k in self.ignore_keys():
                del ret[k]

        return ret

    def _load_results(self) -> dict:
        if not self.result_yaml.exists():
            msg = f"Result YAML not found: {self.result_yaml}"
            raise FileNotFoundError(msg)
        with open(self.result_yaml) as f:
            data = yaml.safe_load(f) or {}
        if "pe_spectrum" not in data:
            msg = "pe_spectrum info not present in result YAML; run analysis first."
            raise RuntimeError(msg)

        return get_physics_object(data, self.ureg)

    def _save_results(self, results: dict[str, dict[int, dict[str, Any]]], key: str) -> None:
        if self.result_yaml.exists():
            with open(self.result_yaml) as f:
                existing = yaml.safe_load(f) or {}
            if key in existing and not self.override_results:
                msg = key + " results already present and override flag is False. Not saving"
                self.logger.error(msg)
                return

            existing[key] = quantity_to_dict(results)
            with open(self.result_yaml, "w") as f:
                yaml.safe_dump(existing, f, default_flow_style=False)
            msg = f"Updated {key} YAML at {self.result_yaml}"
            self.logger.info(msg)

        else:
            with open(self.result_yaml, "w") as f:
                yaml.safe_dump({key: quantity_to_dict(results)}, f, default_flow_style=False)
            self.logger.info("Wrote new result YAML at %s", self.result_yaml)

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
        self._save_results(results, "pe_spectrum")
        self.logger.info("All runs processed.")

    # ----------------------
    # Per-run flow
    # ----------------------
    def analyze_run(self, run_name: str, meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
        # build file paths
        f_raw = self.aux_yaml.parent / Path(
            meta["daq"].replace("daq", "raw").replace("data", "lh5")
        )
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
                ch_idx = int(ch[2:])
                if ch_idx not in meta:
                    self.logger.warning(
                        "Run %s: channel %s (PMT %d) not in aux file, but data exists. Skipping",
                        run_name,
                        ch,
                        ch_idx + 1,
                    )
                    continue

                self.logger.info("Run %s - channel %s (PMT %d)", run_name, ch, ch_idx + 1)
                try:
                    n, bins = self._get_histo(ch, f_dsp)
                    raw_runtime = self._extract_runtime_if_present(f_raw, ch)
                    fig, chan_data = self.process_channel(
                        run_name, ch_idx, meta, n, bins, raw_runtime
                    )
                    # fig may be None if plotting skipped
                    if fig is not None:
                        pdf.savefig(fig)
                        plt.close(fig)
                    run_results[ch_idx] = chan_data
                except Exception as exc:
                    self.logger.exception(
                        "Channel-level error run=%s ch=%s: %s", run_name, ch, exc
                    )
                    run_results[ch_idx] = {"status": "error", "reason": str(exc)}

        self.logger.info("Wrote PDF for run %s to %s", run_name, pdf_path)
        return run_results

    def _get_histo(
        self,
        ch: str,
        f_dsp: Path,
    ) -> tuple[np.NDArray[Any], np.NDArray[Any]]:

        # load data
        try:
            vals = lh5.read_as(f"{ch}/dsp/nnls_solution/values", f_dsp, "np")
        except Exception as e:
            msg = f"Failed to read DSP for {ch} in {f_dsp}: {e}"
            self.logger.warning(msg)
            return None, {"status": "skipped", "reason": msg}

        # compute pe-values
        try:
            pe_vals = np.nansum(np.where(vals > self.lim, vals, np.nan), axis=1)
        except Exception as e:
            msg = f"Failed to compute pe values for {ch}: {e}"
            self.logger.warning(msg)
            return None, {"status": "skipped", "reason": msg}
        return np.histogram(pe_vals, bins=self.bins)

    # ----------------------
    # Per-channel processing
    # ----------------------
    def process_channel(
        self,
        run_name: str,
        ch_idx: int,
        meta: dict[str, Any],
        n: np.NDArray[int],
        bins: np.NDArray[float],
        raw_runtime: float | None = None,
    ) -> tuple[plt.Figure | None, dict[str, Any]]:
        """Process channel. Returns (figure_or_None, channel_result_dict).

        Non-critical failures return a result dict with status 'skipped'
        """
        result: dict[str, Any] = {}
        result["voltage"] = meta[ch_idx]["v10"]

        # histogram
        fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
        ax.stairs(
            values=n,
            edges=bins,
            label=f"channel {ch_idx} (PMT {ch_idx+1}) at {result['voltage'].magnitude:.2f}"
            f" {format(result['voltage'].units,'~')}",
        )

        if self.calib != "None":
            self._add_charge_axis(ax, False)

        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # total counts
        result["statistics"] = {
            "total_cts": self.ureg.Quantity(ufloat(np.sum(n), np.sum(n) ** 0.5), "dimensionless"),
        }

        # runtime
        result["runtime"] = {}
        if "runtime" in meta:
            result["runtime"]["aux"] = meta["runtime"]
        if raw_runtime is not None:
            result["runtime"]["raw"] = raw_runtime * self.ureg.seconds

        # noise-only
        if result["voltage"] <= 10 * self.ureg.volt:
            self._decorate_axis(ax)
            # minimal result (no fit)
            msg = "Voltage a 0 --> Noise run"
            result["status"] = ("skipped",)
            result["reason"] = (msg,)

            return fig, result

        # detect valley & peaks
        vi = valley_index_strict(n)
        if vi is None:
            msg = "Valley detection failed (no strict peak/valley)."
            self.logger.warning("Run %s ch %i: %s", run_name, ch_idx, msg)
            self._decorate_axis(ax)
            result["status"] = ("skipped",)
            result["reason"] = (msg,)
            return fig, result

        noise_peak, valley_idx = vi

        # find first p.e. peak after noise_peak
        sub = n[valley_idx:]
        pe_vi = np.argmax(sub)
        if pe_vi is None:
            msg = "1st-p.e. detection failed after noise peak."
            self.logger.warning("Run %s ch %i: %s", run_name, ch_idx, msg)
            self._decorate_axis(ax)
            return fig, {"status": "skipped", "reason": msg}

        pe_peak_idx = valley_idx + pe_vi

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
            self.logger.warning("Run %s ch %i: %s", run_name, ch_idx, msg)
            self._decorate_axis(ax)
            return fig, {"status": "skipped", "reason": msg}

        amp0 = float(n[pe_peak_idx])
        mu0 = float(bin_centers[pe_peak_idx])
        sigma0 = mu0 - bins[valley_idx]

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
            msg = f"Minuit error for {ch_idx}: {e}"
            self.logger.warning("Run %s ch %i: %s", run_name, ch_idx, msg)
            self._decorate_axis(ax)
            return fig, {"status": "skipped", "reason": msg}

        # Basic validity check
        if not getattr(m, "valid", True):
            self.logger.warning("Minuit invalid result for run %s ch %i", run_name, ch_idx)
            self._decorate_axis(ax)
            return fig, {"status": "skipped", "reason": "Minuit invalid"}

        fit_vals = {k: float(v) for k, v in m.values.to_dict().items()}
        fit_errs = {k: float(v) for k, v in m.errors.to_dict().items()}

        result["pe_peak_fit"] = {
            "mean": self.ureg.Quantity(ufloat(fit_vals["mu"], fit_errs["mu"]), self.ureg.NNLS),
            "sigma": self.ureg.Quantity(
                ufloat(fit_vals["sigma"], fit_errs["sigma"]), self.ureg.NNLS
            ),
            "amp": self.ureg.Quantity(
                ufloat(fit_vals["amp"], fit_errs["amp"]), self.ureg.dimensionless
            ),
        }

        # full fit curve over bins for plotting & integrals
        y_fit = gaussian(bin_centers, fit_vals["amp"], fit_vals["mu"], fit_vals["sigma"])
        ax.plot(bin_centers, y_fit, "r-", label="NLL fit (Minuit)")

        self._decorate_axis(ax)

        # statistics
        try:
            border = 0.75*fit_vals["mu"]
            closest_index = np.abs(bin_centers - border).argmin()

            dcr_idx = np.abs(bin_centers-0.3*fit_vals["mu"]).argmin()
            zero_idx = np.abs(bin_centers).argmin()

            result["statistics"] = {
                "1st_pe_fit_integral_below_valley": self.ureg.Quantity(
                    ufloat(np.sum(y_fit[zero_idx:valley_idx]), np.sum(y_fit[zero_idx:valley_idx]) ** 0.5),
                    "dimensionless",
                ),
                "1st_pe_fit_integral_below_dcr_low_lim": self.ureg.Quantity(
                    ufloat(np.sum(y_fit[zero_idx:dcr_idx]), np.sum(y_fit[zero_idx:dcr_idx]) ** 0.5),
                    "dimensionless",
                ),
                "1st_pe_fit_integral_below_border": self.ureg.Quantity(
                    ufloat(np.sum(y_fit[zero_idx:closest_index]), np.sum(y_fit[zero_idx:closest_index]) ** 0.5),
                    "dimensionless",
                ),
                "cts_above_valley": self.ureg.Quantity(
                    ufloat(np.sum(n[valley_idx:]), np.sum(n[valley_idx:]) ** 0.5), "dimensionless"
                ),
                "cts_above_border": self.ureg.Quantity(
                    ufloat(np.sum(n[closest_index:]), np.sum(n[closest_index:]) ** 0.5), "dimensionless"
                ),
                "1st_pe_fit_integral": self.ureg.Quantity(
                    ufloat(np.sum(y_fit), np.sum(y_fit) ** 0.5), "dimensionless"
                ),
                "total_cts": self.ureg.Quantity(
                    ufloat(np.sum(n), np.sum(n) ** 0.5), "dimensionless"
                ),
                "noise_peak": {
                    "pos": float(bin_centers[noise_peak]) * self.ureg.NNLS,
                    "amp": self.ureg.Quantity(
                        ufloat(n[noise_peak], n[noise_peak] ** 0.5), "dimensionless"
                    ),
                },
                "valley": {
                    "pos": float(bin_centers[valley_idx]) * self.ureg.NNLS,
                    "amp": self.ureg.Quantity(
                        ufloat(n[valley_idx], n[valley_idx] ** 0.5), "dimensionless"
                    ),
                },
                "1st_pe_guess": {
                    "pos": float(mu0) * self.ureg.NNLS,
                    "amp": self.ureg.Quantity(ufloat(amp0, amp0**0.5), "dimensionless"),
                },
            }
        except Exception as e:
            self.logger.warning(
                "Statistics computation failed for run %s ch %i: %s", run_name, ch_idx, e
            )
            result.setdefault("statistics", {})
            result["statistics"]["error"] = str(e)

        result["status"] = "ok"
        return fig, result

    # ----------------------
    # Utilities
    # ----------------------
    def _decorate_axis(self, ax: plt.Axes) -> None:
        ax.set_xlim(-10, self.bins[-1])
        ax.set_ylim(0.5, None)
        ax.set_yscale("log")
        ax.set_ylabel(f"Counts/{self.bin_size} NNLS")
        ax.set_xlabel("Charge (NNLS)")
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

    def _add_charge_axis(self, ax: plt.Axes, is_y) -> None:
        """
        This function adds a axis in new Charge units to a given axis plot.
        if the unit is not charge convertible (or is gain) will log a warning
        and fall gracefully.

        Parameters
        ----------
        ax : plt.Axes
            The plot to which to add the new axis.
        is_y : bool
            If True add a new y-axis. Else an x-axis.
        """

        if self.calib == "gain":
            echarge = 1.60217663E-19
            label = "Gain (a.u.)"
            func = (
                lambda x: ((x * self.ureg.NNLS).to("elementary_charge").m),
                lambda y: (y * self.ureg.elementary_charge).to("NNLS").m,
            )
        else:
            if not self.ureg.NNLS.is_compatible_with(self.calib):
                msg = f"Unit [{self.calib}] not compatible with charge"
                self.logger.warning(msg)
            label = f"Charge ({self.calib})"
            func = (
                lambda x: (x * self.ureg.NNLS).to(self.calib).m,
                lambda y: (y * self.ureg(self.calib)).to("NNLS").m,
            )

        if is_y:
            secax_y = ax.secondary_yaxis("right", functions=func)
            secax_y.set_ylabel(label)
        else:
            secax_x = ax.secondary_xaxis("top", functions=func)
            secax_x.set_xlabel(label)

    def _unit_converter(self, v, target_unit, constant=1):
        """
        Take a value v and if its a Quantity
        apply conversion of the targeted unit.
        A constant can be multiplied.
        E.g.:
        target unit = pC
        v = 2.5e-24 C**2/V*m
        returns 2.5 pC**2/V/m
        """
        if not isinstance(v, self.ureg.Quantity):
            return v
        dims = dict(v._units)
        # Decompose units
        dims = dict(v._units)
        target_units = {
            k: d for k, d in dims.items() if self.ureg(k).is_compatible_with(target_unit)
        }
        other_units = {
            k: d for k, d in dims.items() if not self.ureg(k).is_compatible_with(target_unit)
        }

        # Start with dimensionless
        v_base = v / v.units

        # Multiply back non-target units
        for k, d in other_units.items():
            v_base *= self.ureg(k) ** d

        # Convert target units to target unit
        for k, d in target_units.items():
            v_base *= (self.ureg(k).to(target_unit)) ** d

        return v_base * constant

    # ----------------------
    # Signal to Noise Ratio (SNR)
    # ----------------------
    def compute_snr(self) -> None:
        data = self._load_results()
        snr = {}
        for run_name, pmt_dict in data["pe_spectrum"].items():
            run_snr = {}
            for pmt, info in pmt_dict.items():
                try:
                    if info.get("voltage") < CONSIDERED_OFF_VOLTAGE * self.ureg.V:
                        msg = (
                            f"PMT {pmt} at {format(info.get('voltage'),'~.2f')} is considered off."
                        )
                        self.logger.warning(msg)
                        continue
                    noise = info.get("statistics").get("valley").get("amp")

                    if info.get("status", "skipped") == "ok":
                        signal = info.get("pe_peak_fit").get("amp")
                    else:
                        signal = info.get("statistics").get("1st_pe_guess").get("amp")
                    run_snr[pmt] = signal / noise
                except Exception as e:
                    self.logger.warning(
                        "Failed to compute SNR for run %s channel %s: %s", run_name, pmt, e
                    )
            snr[run_name] = run_snr

        self._save_results(snr, "snr")

        fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
        for run_name, pmt_dict in snr.items():
            temp = self.aux.get(run_name).get("pth_start", {}).get("temperature")
            pmts = sorted(pmt_dict.keys())
            pmts = sorted(pmt_dict.keys())
            vals = [pmt_dict[p].n for p in pmts]
            errs = [pmt_dict[p].s for p in pmts]
            lbl = self.aux.get(run_name).get("description", None)
            if lbl is None:
                lbl = run_name
            lbl += f" ({format(temp,'~.2f')})"
            ax.errorbar(pmts, vals, errs, label=lbl, fmt="o")
        ax.set_xlabel("Channel")
        ax.set_ylabel("SNR (a.u.)")
        ax.set_title("SNR per Channel (= peak / valley)")
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
        data = self._load_results()
        dcr = {}
        for run_name, pmt_dict in data["pe_spectrum"].items():
            run_dcr = {}
            for pmt, info in pmt_dict.items():
                try:
                    if info.get("voltage") < CONSIDERED_OFF_VOLTAGE * self.ureg.volt:
                        msg = (
                            f"PMT {pmt} at {format(info.get('voltage'),'~.2f')} is considered off."
                        )
                        self.logger.warning(msg)
                        continue
                    stats = info.get("statistics", {})
                    runtime_info = info.get("runtime", {})
                    if time_mode not in runtime_info:
                        self.logger.warning(
                            "Run %s channel %s: missing runtime '%s'; skipping DCR.",
                            run_name,
                            pmt,
                            time_mode,
                        )
                        continue
                    dcts = stats.get("1st_pe_fit_integral_below_border") + stats.get("cts_above_border") - stats.get("1st_pe_fit_integral_below_dcr_low_lim")
                    runtime = runtime_info[time_mode]
                    run_dcr[pmt] = dcts / runtime

                except Exception as e:
                    self.logger.warning(
                        "Failed to compute DCR for run %s channel %s: %s", run_name, pmt, e
                    )
            dcr[run_name] = run_dcr
        self._save_results(dcr, "dcr")

        fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
        for run_name, pmt_dict in dcr.items():
            temp = self.aux.get(run_name).get("pth_start", {}).get("temperature")
            pmts = sorted(pmt_dict.keys())
            vals = [pmt_dict[p].n for p in pmts]
            errs = [pmt_dict[p].s for p in pmts]
            lbl = self.aux.get(run_name).get("description", None)
            if lbl is None:
                lbl = run_name
            lbl += f" ({format(temp,'~.2f')})"
            ax.errorbar(pmts, vals, errs, label=lbl, fmt="o")
        ax.set_xlabel("Channel")
        ax.set_ylabel("DCR (Hz)")
        ax.set_title("Dark Count Rate per Channel")
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
        data = self._load_results()
        tmp_dic = {"used_keys": [], "temperatures": []}

        if len(data["pe_spectrum"]) < 2:
            msg = f"Only found {len(data['pe_spectrum'])} datapoints <2 in {self.result_yaml}"
            self.logger.error(msg)
            return
        if len(data["pe_spectrum"]) < 3:
            msg = f"Only found 2 datapoints in {self.result_yaml}. Fit will have no uncertainties."
            self.logger.warning(msg)

        for key, run in data["pe_spectrum"].items():
            tmp_dic["used_keys"].append(key)
            tmp_dic["temperatures"].append(
                self.aux.get(key).get("pth_start", {}).get("temperature")
            )
            pop_counter = 0
            for pmt in run:
                v = run[pmt].get("voltage", 0.0 * self.ureg.volts)
                if v < CONSIDERED_OFF_VOLTAGE * self.ureg.volts:
                    msg = f"PMT {pmt} at {format(v,'~.2f')} is considered off."
                    self.logger.warning(msg)
                    pop_counter += 1
                    continue
                if pmt not in tmp_dic:
                    tmp_dic[pmt] = {"voltage": [], "vals": []}
                tmp_dic[pmt]["voltage"].append(v)
                tmp_dic[pmt]["vals"].append(run[pmt]["pe_peak_fit"]["mean"])
            if pop_counter == len(run):
                tmp_dic["used_keys"].pop()
                tmp_dic["temperatures"].pop()

        pdf_path = self.plot_folder / "gain_plots.pdf"
        with PdfPages(pdf_path) as pdf:
            temp_str = ", ".join([format(t, "~.2f") for t in tmp_dic["temperatures"]])
            for key, pmt in tmp_dic.items():
                if key in ["used_keys", "temperatures"]:
                    continue
                fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
                xunit = pmt["voltage"][0].units
                pmt["voltage"] = [i.to(xunit) for i in pmt["voltage"]]
                yunit = pmt["vals"][0].units
                pmt["vals"] = [i.to(yunit) for i in pmt["vals"]]
                ax.errorbar(
                    [i.m for i in pmt["voltage"]],
                    [i.n for i in pmt["vals"]],
                    [i.s for i in pmt["vals"]],
                    label=f"Channel {key} ({temp_str})",
                    fmt="o",
                )
                ax.set_xlabel(f"Voltage ({format(xunit,'~')})")
                ax.set_ylabel(f"Gain ({format(yunit,'~')})")

                if self.calib != "None":
                    self._add_charge_axis(ax, True)

                params, covariance = curve_fit(
                    f=linear_func,
                    xdata=[i.m for i in pmt["voltage"]],
                    ydata=[i.n for i in pmt["vals"]],
                    sigma=[i.s for i in pmt["vals"]],
                    absolute_sigma=True,
                )
                a_opt, b_opt = params
                perr = np.sqrt(np.diag(covariance))
                x = np.linspace(-1 * b_opt / a_opt, max([i.m for i in pmt["voltage"]]) + 10, 1000)
                ax.plot(x, linear_func(x, a_opt, b_opt), ls="--", color="red", label="Fit")
                tmp_dic[key]["a"] = self.ureg.Quantity(ufloat(a_opt, perr[0]), yunit / xunit)
                tmp_dic[key]["b"] = self.ureg.Quantity(ufloat(b_opt, perr[1]), yunit)

                tmp_dic[key]["func"] = "G = a*voltage+b"
                pmt.pop("voltage")
                pmt.pop("vals")
                ax.legend()
                pdf.savefig(fig)
                plt.close(fig)

        self._save_results(tmp_dic, "linear_gain")

    def calibrate_nnls_values(self, output_file: str):
        """
        Reads the current results.yaml,
        finds all entries that contain a Charge compatible quantities
        Applies charge conversion as per self.calib
        writes out calibrated file to output_file.
        """
        if self.calib != "gain" and not self.ureg.NNLS.is_compatible_with(self.calib):
            msg = f"Unit [{self.calib}] not compatible with charge"
            self.logger.error(msg)
            return

        data = self._load_results()

        def convert_charge_units(d):
            """Recursively convert all Quantities compatible with charge to new_unit."""

            for k, v in d.items():
                if isinstance(v, dict):
                    convert_charge_units(v)  # recurse
                elif isinstance(v, self.ureg.Quantity):
                    if self.calib == "gain":
                        d[k] = self._unit_converter(v, self.ureg.elementary_charge)#, 1./(1.60217663E-19*self.ureg("C")) )
                    else:
                        d[k] = self._unit_converter(v, self.calib)

        # Convert all charge-compatible quantities to nanocoulombs
        convert_charge_units(data)

        try:
            with open(output_file, "w") as f:
                yaml.safe_dump(quantity_to_dict(data), f, default_flow_style=False)
        except Exception as e:
            msg = f"Failed to write calibrated file: {e}"
            self.logger.error(msg)
            raise

        msg = f"Calibrated results written to {output_file}"
        self.logger.info(msg)
