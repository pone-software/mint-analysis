from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from icecube import icetray
from matplotlib.backends.backend_pdf import PdfPages

from mintanalysis.pmt.ana.peSpectrumAnalyzer import PESpectrumAnalyzer


class IceTraySpectrumAnalyzer(PESpectrumAnalyzer):
    def __init__(
        self,
        aux_yaml,
        f_i3: Path | None = None,
        keys=None,
        ignore_keys: list | None = None,
        bin_size: int = 100,
        lim=0,
        override_results=False,
        logger=None,
        calibrator=None,
        calib="None",
    ):
        super().__init__(
            aux_yaml, keys, ignore_keys, bin_size, lim, override_results, logger, calibrator, calib
        )
        self.bins = np.arange(0, 10, 10.0 / bin_size)
        self.f_i3 = f_i3
        icetray.set_log_level(icetray.I3LogLevel.LOG_WARN)

    def _set_up_paths(self):
        self.plot_folder = self.aux_yaml.parent / "../ana/i3_plots"
        self.result_yaml = self.aux_yaml.parent / "../ana/i3_results.yaml"
        self.plot_folder.mkdir(parents=True, exist_ok=True)

    def analyze_run(self, run_name, meta):
        # build file paths
        if self.f_i3 is None:
            f_i3 = self.aux_yaml.parent / Path(
                meta["daq"].replace("daq", "i3").replace("data", "i3")
            )
        else:
            f_i3 = self.f_i3
        if not f_i3.exists():
            msg = f"Raw file for run {run_name} not found: {f_i3}"
            raise FileNotFoundError(msg)

        run_results: dict[int, dict[str, Any]] = {}
        pdf_path = self.plot_folder / f"pe_spectra_{run_name}.pdf"

        charge_counts = {}

        def histogram_charges(frame, charge_counts, pulses_name="UnfoldedPulses"):

            if not frame.Has(pulses_name):
                return
            pulses = frame.Get(pulses_name)

            for channel, ch_pulses in pulses.items():
                if channel.pmt not in charge_counts:
                    charge_counts[channel.pmt] = []

                charge = 0
                for pulse in ch_pulses:
                    if pulse.charge > self.lim:
                        charge += pulse.charge
                charge_counts[channel.pmt].append(charge)

        tray = icetray.I3Tray()
        tray.AddModule("I3Reader", "reader", filename=str(f_i3))
        tray.AddModule(
            histogram_charges, charge_counts=charge_counts, Streams=[icetray.I3Frame.Physics]
        )
        tray.Execute()
        tray.Finish()

        # collect figures and write once
        with PdfPages(pdf_path) as pdf:
            for ch_idx, v in sorted(charge_counts.items()):
                if ch_idx not in meta:
                    self.logger.warning(
                        "Run %s: channel %s (PMT %d) not in aux file, but data exists. Skipping",
                        run_name,
                        ch_idx,
                        ch_idx + 1,
                    )
                    continue

                self.logger.info("Run %s - channel %s (PMT %d)", run_name, ch_idx, ch_idx + 1)
                try:
                    n, bins = np.histogram(v, bins=self.bins)
                    raw_runtime = self._extract_runtime_if_present(str(f_i3))
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
                        "Channel-level error run=%s ch=%s: %s", run_name, ch_idx, exc
                    )
                    run_results[ch_idx] = {"status": "error", "reason": str(exc)}

        self.logger.info("Wrote PDF for run %s to %s", run_name, pdf_path)
        return run_results

    def _extract_runtime_if_present(self, f_i3):
        times = []

        def get_runtime(frame, times):
            if not frame.Has("I3EventHeader"):
                return
            # Assume here all data is taken in the same year
            times.append(frame.Get("I3EventHeader").start_time.utc_daq_time)

        tray = icetray.I3Tray()
        tray.AddModule("I3Reader", "reader", filename=str(f_i3))
        tray.AddModule(get_runtime, times=times, Streams=[icetray.I3Frame.DAQ])
        tray.Execute()
        tray.Finish()
        return (max(times) - min(times)) * 1e-10

    def _decorate_axis(self, ax: plt.Axes) -> None:
        xmin = np.min(self.bins)
        xmin = xmin - 0.05 * xmin
        xmax = np.max(self.bins)
        xmax = xmax + 0.05 * xmin
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.5, None)
        ax.set_yscale("log")
        ax.set_ylabel(f"Counts/{self.bin_size} NNLS")
        ax.set_xlabel("Charge (NNLS)")
        ax.set_title(f"I3 reconstruction ({self.lim} units solution cut-off)")
        ax.legend()
