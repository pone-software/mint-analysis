import argparse
import copy
import logging
import os

from icecube import icetray, pone_unfolding

from mintanalysis.pmt.ana.utils import setup_logging
from mintanalysis.pmt.i3.funny_heartbeat import heartbeat_ctx


class injectEmptyPhysics(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.ctr = 0

    def Process(self):
        self.current_frame = self.PopFrame()
        if self.current_frame.Stop == icetray.I3Frame.DAQ:
            if self.ctr >= 1:
                self.Inject()
            self.ctr += 1
        self.PushFrame(self.current_frame)

    def Inject(self):
        # Inject empty physics frame
        frame = icetray.I3Frame(icetray.I3Frame.Physics)
        self.PushFrame(frame)


class WaveFormUnfold(pone_unfolding.WaveUnfold):
    def DAQ(self, frame):
        self.PushFrame(frame)

    def Physics(self, frame):
        ch_check_key = "P1ChannelCalibration"
        channels = frame[ch_check_key].keys()

        # Delete waveforms for channels not currently being used
        raw_data = copy.copy(frame["RawData"])
        for c in [c for c in raw_data if c not in channels]:
            del raw_data[c]
        frame.Delete("RawData")
        frame.Put("RawData", raw_data, icetray.I3Frame.Stream("Q"))

        super().DAQ(frame)


def main():
    parser = argparse.ArgumentParser(description="Build P frame.")
    parser.add_argument(
        "-i",
        "--f_i3_in",
        help="Path to  input i3 file",
    )
    parser.add_argument(
        "-o",
        "--f_i3_out",
        default=None,
        help="Path to  output i3 file (needed if append flag is not set)",
    )
    parser.add_argument("-a", "--append", action="store_true", help="Append existing i3 file")
    parser.add_argument(
        "-u", "--upsample", type=int, default=100, help="Upsample factor (spe per bin)"
    )
    parser.add_argument(
        "-t", "--tolerance", type=float, default=2, help="Chi^2 stopping tolerance"
    )
    parser.add_argument(
        "-w",
        "--waveform",
        default="",
        help="Name of refolded waveform field (no waveforms stored if omitted or empty string)",
    )

    args = parser.parse_args()

    if not args.append:
        if args.f_i3_out is None:
            msg = "Output not set. Either use append flag or specify output path."
            raise ValueError(msg)
        if os.path.isfile(args.f_i3_out):
            msg = "Output file exists."
            raise ValueError(msg)

    i3_split = args.f_i3_in.split(".")
    f_i3_out = i3_split[0] + "_aux." + "".join(i3_split[1:]) if args.append else args.f_i3_out
    f_log = f_i3_out.split(".")[0] + ".log"

    logger = setup_logging(log_file=f_log, level=logging.INFO)

    msg = "Creating P frame. This may take a while"
    logger.info(msg)
    with heartbeat_ctx():
        tray = icetray.I3Tray()
        tray.AddModule("I3Reader", "reader", filename=args.f_i3_in)
        tray.AddModule(injectEmptyPhysics)
        tray.AddModule(
            WaveFormUnfold,
            SPEsPerBin=args.upsample,
            OutputWaveforms="",
            RefoldedWaveforms=args.waveform,
            Tolerance=args.tolerance,
        )
        tray.AddModule("I3Writer", "writer", Filename=f_i3_out)
        tray.Execute()
        tray.Finish()

    if args.append:
        os.rename(f_i3_out, args.f_i3_in)

    msg = f"Finished processing. i3 file {f_i3_out if not args.append else args.f_i3_in} generated"
    logger.info(msg)


if __name__ == "__main__":
    main()
