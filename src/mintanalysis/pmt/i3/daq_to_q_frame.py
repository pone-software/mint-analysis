import argparse
import logging
import os

from icecube import icetray, pone_unfolding, dataio

from mintanalysis.pmt.ana.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Build Q frame from DAQ data.")
    parser.add_argument(
        "-i",
        "--f_i3",
        default=None,
        help="Path to i3 file (if omitted replaces all occurrences of daq in f_daq with i3)",
    )
    parser.add_argument("-d", "--f_daq", help="Path to DAQ file", required=True)
    parser.add_argument(
        "-n",
        "--n_events",
        default=0,
        type=int,
        help="Number of events to process (omit or 0 for all)",
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Override existing output file"
    )

    args = parser.parse_args()

    daq_ext = args.f_daq.split(".")[-1]
    f_i3 = (
        args.f_daq.replace("daq", "i3").replace(daq_ext, "i3") if args.f_i3 is None else args.f_i3
    )

    # Create raw folders if not existing
    dir = os.path.dirname(f_i3)
    if dir:
        os.makedirs(dir, exist_ok=True)

    f_log = f_i3.replace(f_i3.split(".")[-1], "log")
    logger = setup_logging(log_file=f_log, level=logging.INFO)

    # Check if I3 file exists and if overwrite flag is set
    # if so move it to back-up file (which will be deleted at the end)
    if os.path.isfile(f_i3):
        if args.overwrite:
            os.rename(f_i3, f_i3 + ".bak")
        else:
            msg = f"I3 file {f_i3} exists and overwrite flag is not specified."
            logger.error(msg)
            return

    # Generate i3 file
    msg = f"Start building i3 file from DAQ file {args.f_daq}"
    logger.info(msg)
    tray = icetray.I3Tray()
    tray.AddModule(pone_unfolding.P1Reader, Input=args.f_daq, OM=1)
    tray.AddModule("I3Writer", "writer", Filename=f_i3)
    if args.n_events > 0:
        tray.Execute(args.n_events)
    else:
        tray.Execute()
    tray.Finish()
    msg = f"Finished processing. i3 file {f_i3} generated"
    logger.info(msg)

    if os.path.isfile(f_i3 + ".bak"):
        os.remove(f_i3 + ".bak")


if __name__ == "__main__":
    main()
