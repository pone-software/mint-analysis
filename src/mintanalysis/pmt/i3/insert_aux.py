import argparse
import logging
import os
import re
from pathlib import Path

import yaml
from icecube import dataclasses, icetray, p1_dataclasses
from pint import UnitRegistry

from mintanalysis.pmt.ana.utils import get_physics_object, setup_logging


class injectDetectorInfo(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter("config", "Aux dictionary of Module")
        self.AddParameter("scale", "ADC scaling factor in mV (default: 0.25 mV)", 0.25)
        self.ctr = True

    def Configure(self):
        self.config = self.GetParameter("config")
        self.scale = self.GetParameter("scale")

    def Process(self):
        self.current_frame = self.PopFrame()
        if self.current_frame.Stop == icetray.I3Frame.DAQ and self.ctr:
            self.Inject()
            self.ctr = False
        self.PushFrame(self.current_frame)

    def Inject(self):
        # Inject Aux data (geometry frame)
        frame = icetray.I3Frame(icetray.I3Frame.Geometry)
        if "channels" not in self.config:
            return
        channels = [icetray.OMKey(0, 1, i) for i in self.config["channels"]]
        for k, v in self.config.items():
            if isinstance(v, list):
                map = dataclasses.I3MapKeyDouble()
                for i in range(len(channels)):
                    map[channels[i]] = v[i]
                if k == "v10_in_V":
                    frame.Put("PMTVoltages", map)
                else:
                    frame.Put(k, map)
            else:
                frame.Put(k, map)

        # Inject ChannelCalibrationMap (calibration frame)
        cal = p1_dataclasses.P1ChannelCalibrationMap()
        for ch in channels:
            ccal = p1_dataclasses.P1ChannelCalibration()
            ccal.SetBaseline(frame["channel_means"][ch])
            ccal.SetVoltageFactor(self.scale * icetray.I3Units.millivolt)
            ccal.SetNoiseLevel(frame["noise_levels"][ch])
            cal[ch] = ccal

        self.PushFrame(frame)
        frame = icetray.I3Frame(icetray.I3Frame.Calibration)
        frame.Put("P1ChannelCalibration", cal)
        self.PushFrame(frame)


def load_aux(aux_yaml: Path, key: str, ch_mask: int = 0xFFFF) -> dict:
    # get list of channels from channel mask
    channels = []
    for i in range(16):
        bit_value = 1 << i
        if ch_mask & bit_value:
            channels.append(i)

    ureg = UnitRegistry()
    if not aux_yaml.exists():
        msg = f"Aux file not found: {aux_yaml}"
        raise FileNotFoundError(msg)
    with open(aux_yaml) as f:
        aux = yaml.safe_load(f)

    # convert to physics units
    aux = get_physics_object(aux, ureg)
    aux = aux[key]

    ret = {"channels": [], "channel_thresholds": [], "channel_means": [], "noise_levels": []}

    for i in range(16):
        if (i in aux) and (i in channels):
            ret["channels"].append(i)
            ret["channel_thresholds"].append(
                aux.get("collector_config", {}).get("ChannelThresholds", [])[i]
            )
            ret["channel_means"].append(aux.get("collector_config", {}).get("ChannelMeans", [])[i])
            ret["noise_levels"].append(
                (ret["channel_thresholds"][-1] - ret["channel_means"][-1]) / 5
            )  # assumption: threshold is set at 5 sigma

            # make sure values are in desired units
            for p in [("v10", "V"), ("vsup", "V"), ("di10", "mA"), ("isup", "mA")]:
                if p[0] in aux[i]:
                    k = f"{p[0]}_in_{p[1]}"
                    if k not in ret:
                        ret[k] = []
                    ret[k].append(aux[i][p[0]].to(p[1]).magnitude)

    if "runtime" in aux:
        ret["runtime_in_s"] = aux["runtime"].to("s").magnitude

    for pth in ["pth_start", "pth_end"]:
        if pth in aux:
            for v in [
                ("humidity", "%", "percent"),
                ("pressure", "Pa"),
                ("temperature", "degree_Celsius"),
            ]:
                if v[0] in aux[pth]:
                    ret[f"{pth.split('_')[-1]}_{v[0]}_in_{v[1] if len(v) == 2 else v[2]}"] = (
                        aux[pth][v[0]].to(v[1]).magnitude
                    )

    return ret


def main():
    parser = argparse.ArgumentParser(description="Build G frame from aux data.")
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
    parser.add_argument(
        "-a",
        "--f_aux",
        help="Path to aux file",
    )
    parser.add_argument(
        "-k",
        "--key",
        default=None,
        help="run key (if omitted key is inferred from i3 file name)",
    )
    parser.add_argument("-e", "--expand", action="store_true", help="Expand existing i3 file")
    parser.add_argument("-c", "--channel_mask", default=0xFFFF, type=int, help="Channel mask")
    parser.add_argument(
        "-s",
        "--scale",
        default=0.25,
        type=float,
        help="ADC scaling factor in mV (default 0.25 mV)",
    )

    args = parser.parse_args()

    if not args.expand:
        if args.f_i3_out is None:
            msg = "Output not set. Either use append flag or specify output path."
            raise ValueError(msg)
        if os.path.isfile(args.f_i3_out):
            msg = "Output file exists."
            raise ValueError(msg)

    i3_split = args.f_i3_in.split(".")
    f_i3_out = i3_split[0] + "_aux." + "".join(i3_split[1:]) if args.expand else args.f_i3_out
    f_log = f_i3_out.split(".")[0] + ".log"

    logger = setup_logging(log_file=f_log, level=logging.INFO)

    # if key is None see if we can find it in the path of the aux file
    # we are looking for YYYY_MM_DD_HH_MM_SS
    key = args.key
    if key is None:
        match = re.search(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", args.f_i3_in)
        if match is not None:
            key = match.group()
        else:
            msg = "Key not provided and can not infer from aux path"
            raise ValueError(msg)

    # Generate i3 file
    msg = f"Writing aux info to G frame {args.f_aux}"
    logger.info(msg)
    tray = icetray.I3Tray()
    tray.AddModule("I3Reader", "reader", filename=args.f_i3_in)
    tray.AddModule(
        injectDetectorInfo,
        config=load_aux(Path(args.f_aux), key=key, ch_mask=args.channel_mask),
        scale=args.scale,
    )
    tray.AddModule("I3Writer", "writer", Filename=f_i3_out)
    tray.Execute()
    tray.Finish()

    if args.expand:
        os.rename(f_i3_out, args.f_i3_in)

    msg = f"Finished processing. i3 file {f_i3_out if not args.expand else args.f_i3_in} generated"
    logger.info(msg)


if __name__ == "__main__":
    main()
