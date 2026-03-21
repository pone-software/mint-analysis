from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .utils import setup_logging

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
        help="Choose a charge calibration unit",
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
    parser.add_argument(
        "-i",
        "--icetray",
        action="store_true",
        help="Analysis of i3 datastream (Otherwise pygama datastream)",
    )
    parser.add_argument(
        "-k",
        "--keys",
        nargs="+",
        default=None,
        help="Only analyse this keys, ignore all other keys in aux file",
    )
    parser.add_argument(
        "--ignore_keys",
        nargs="+",
        default=None,
        help="Ignore these keys in aux file",
    )

    args = parser.parse_args()

    if args.icetray:
        f_log = Path(args.aux_file).parent / "../ana/i3_analysis.log"
    else:
        f_log = Path(args.aux_file).parent / "../ana/analysis.log"

    f_log.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_file=f_log, level=logging.INFO)
    try:
        if args.icetray:
            from mintanalysis.pmt.ana.peSpectrumAnalyzerIcetray import IceTraySpectrumAnalyzer

            analyzer = IceTraySpectrumAnalyzer(
                logger=logger,
                aux_yaml=Path(args.aux_file),
                keys=args.keys,
                ignore_keys=args.ignore_keys,
                bin_size=args.bin_size,
                lim=args.nnls_limit,
                override_results=args.override,
                calib=args.calibrate,
            )
        else:
            from mintanalysis.pmt.ana.peSpectrumAnalyzer import PESpectrumAnalyzer

            analyzer = PESpectrumAnalyzer(
                logger=logger,
                aux_yaml=Path(args.aux_file),
                keys=args.keys,
                ignore_keys=args.ignore_keys,
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
            analyzer.compute_snr()
        if args.calibrate != "None":
            analyzer.calibrate_nnls_values(
                output_file=str(analyzer.result_yaml).replace(".yaml", "_calibrated.yaml")
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
