import argparse
import logging
import os

import awkward as ak
import numpy as np
import tqdm

# LEGEND specific imports
from lgdo import (
    Array,
    ArrayOfEqualSizedArrays,
    Table,
    VectorOfVectors,
    WaveformTable,
    lh5,
)


def timestamps_sec_ps(
    start_seconds: int, ticks: np.ndarray, start_ps: int = 0, tick_unit: int = 4800
):
    """
    Convert a start time and tick count (in tick units ) into a timestamp
    returned as sec and ps int64 numpy arrays.

    Parameters
    ----------
    start_seconds : int
        Integer seconds for the start time.
    ticks : numpy array
        Array of ticks since the start (in tick_units).
    start_ps : int, optional
        Additional offset at the start in picoseconds.
    tick_unit : int, optional
        Additional offset at the start in picoseconds.

    Returns
    -------
    two numpy int64 arrays: seconds, picoseconds
    """

    # Use object dtype to prevent overflow during computation
    ticks = np.asarray(ticks, dtype=object)
    total_ps = start_seconds * 10**12 + start_ps + ticks * tick_unit
    vdivmod = np.vectorize(lambda x: divmod(x, 10**12), otypes=[object, object])
    sec, ps = vdivmod(total_ps)

    return sec.astype(np.int64), ps.astype(np.int64)


def build_raw(
    f_daq: str,
    f_raw: str,
    jagged: bool = False,
    ticks_value_ns: float = 4.8,
    trigger_offset: int = 0,
    verbose: bool = False,
    logger: logging.Logger | None = None,
):
    """
    Decode fastDAQ eng. format to lh5

    Parameters
    ----------
    f_daq : str
        Path to DAQ file.
    f_raw : str
        Path to raw file
    jagged : bool, optional
        Set to true if waveform lengths are not all same length
    ticks_value_ns : float, optional
        ADC sampling width in nanoseconds.
    trigger_offset : int, optional
        Trigger offset from 0 (in ADC samples).
    verbose : bool, optional
        Turn verbosity on.
    """
    msg = f"Start processing file {f_daq}"
    logger.info(msg)
    with open(f_daq) as f:
        lines = f.readlines()

    # Parse metadata
    logger.info("Parsing metadata")
    metadata = {}
    i = 0
    for line in lines:
        if line.startswith("Trigger"):
            break
        i += 1
        if line.startswith("#") or line.isspace():
            continue
        key, *vals = line.split()
        metadata[key] = vals if len(vals) > 1 else vals[0]

    if "TimeSync" in metadata and len(metadata["TimeSync"]) in [3, 4]:
        if len(metadata["TimeSync"]) == 4:  # old fastDAQ version
            metadata["start_in_utc_seconds"] = int(metadata["TimeSync"][0], 16) + int(
                metadata["TimeSync"][1], 16
            )
        elif len(metadata["TimeSync"]) == 3:
            metadata["start_in_utc_seconds"] = int(metadata["TimeSync"][0], 16)

        metadata["start_in_ps_after_seconds"] = int(metadata["TimeSync"][-2], 16)
        metadata["deviation_to_sysref_in_ps"] = int(metadata["TimeSync"][-1], 16)
        metadata.pop("TimeSync")

    lines = lines[i:]

    triggers = []
    current_trigger = None
    rec_lenth = -1
    ch = None

    msg = f"Processing body with {len(lines)} lines"
    logger.info(msg)
    for li in tqdm.tqdm(lines, desc="Processing file", disable=(not verbose)):
        line = li.strip()
        if not line:
            continue
        parts = line.split()

        if parts[0] == "Trigger":
            if (
                current_trigger is not None
                and ch is not None
                and len(current_trigger["waveforms"].keys()) > 0
                and (jagged or len(current_trigger["waveforms"][ch]) == rec_lenth)
            ):
                triggers.append(current_trigger)

            current_trigger = {
                "mask": parts[1],
                "timestamp": int(parts[2]),
                "delta": int(parts[3]) if len(parts) > 3 else None,
                "waveforms": {},
            }

        elif parts[0] == "Waveform":
            ch = int(parts[1])
            nsamp = int(parts[2])
            samples = np.array(list(map(int, parts[3:])), dtype=np.int16)
            if len(samples) != nsamp:
                msg = f"Channel {ch}: expected {nsamp} samples, got {len(samples)}"
                raise ValueError(msg)
            current_trigger["waveforms"][ch] = samples
            if rec_lenth == -1:
                rec_lenth = nsamp

    if (current_trigger is not None) and (
        jagged or len(current_trigger["waveforms"][ch]) == rec_lenth
    ):
        triggers.append(current_trigger)

    # ---------------------------
    # Reorganize: channel triggers
    # ---------------------------
    logger.info("Reorganizing event structure")
    all_channels = sorted({ch for trig in triggers for ch in trig["waveforms"]})
    channels = {ch: [] for ch in all_channels}
    carried_deltas = dict.fromkeys(all_channels, 0)

    for trig in tqdm.tqdm(triggers, desc="Reorganizing event structure", disable=(not verbose)):
        delta = trig["delta"] or 0
        for ch in all_channels:
            if ch in trig["waveforms"]:
                channels[ch].append(
                    {
                        "waveform": trig["waveforms"][ch],
                        "timestamp": trig["timestamp"],
                    }
                )
                carried_deltas[ch] = 0
            else:
                carried_deltas[ch] += delta

    # ---------------------------
    # Convert to LH5 via Awkward
    # ---------------------------
    logger.info("Converting to LH5")
    for ch in tqdm.tqdm(all_channels, desc="Convert to LH5", disable=(not verbose)):
        akdata = ak.Array(channels[ch])
        if jagged:
            a = VectorOfVectors(data=akdata.waveform)
        else:
            a = ArrayOfEqualSizedArrays(
                nda=ak.to_numpy(akdata.waveform, allow_missing=False).astype(np.int16)
            )
        b = WaveformTable(
            values=a,
            dt=ticks_value_ns,
            dt_units="ns",
            t0=trigger_offset * ticks_value_ns,
            t0_units="ns",
        )

        table = Table(size=len(b))
        table.add_field("waveform", b, True)

        # Convert timestamp into a more convenient format for analysis
        if "start_in_utc_seconds" in metadata:
            sec, ps = timestamps_sec_ps(
                metadata["start_in_utc_seconds"],
                ak.to_numpy(akdata.timestamp, allow_missing=False),
                metadata.get("start_in_ps_after_seconds", 0),
                round(ticks_value_ns * 1000),
            )
            sec = Array(sec)
            sec.attrs["units"] = "s"

            ps = Array(ps)
            ps.attrs["units"] = "ps"

            table.add_field("timestamp_sec", sec, True)
            table.add_field("timestamp_ps", ps, True)

        else:
            c = Array(ak.to_numpy(akdata.timestamp, allow_missing=False))
            table.add_field("timestamp", c, True)

        lh5.write(table, name="raw", group=f"ch{ch:03}", lh5_file=f_raw)

    msg = f"Done! Raw tier created at {f_raw}"
    logger.info(msg)


# CLI entry point
def main():
    parser = argparse.ArgumentParser(description="Build raw tier from DAQ input.")
    parser.add_argument("-r", "--f_raw", help="Path to raw file", required=True)
    parser.add_argument("-d", "--f_daq", help="Path to DAQ file", required=True)
    parser.add_argument("-t", "--tick_value", default=4.8, help="Tick value in ns")
    parser.add_argument("-s", "--shift", default=0, help="Shift of trigger position in samples")
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Override existing output file"
    )
    parser.add_argument("-j", "--jagged", action="store_true", help="DAQ data is jagged")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    args = parser.parse_args()

    logger = logging.getLogger("daq2raw")
    log_level = logging.INFO
    logger.setLevel(log_level)

    fmt = logging.Formatter("[%(asctime)s] [%(name)s - %(funcName)s] [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_file = args.f_raw.replace(args.f_raw.split(".")[-1], "log")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if os.path.exists(args.f_raw):
        if args.overwrite:
            try:
                os.remove(args.f_raw)
                msg = f"Deleting old file at {args.f_raw}"
                logger.info(msg)
            except PermissionError:
                msg = f"Permission denide to delete {args.f_raw}"
                logger.error(msg)
            except Exception as e:
                msg = f"An error occurred while deleting {args.f_raw}: {e}"
                logger.error(msg)
        else:
            msg = f"{args.f_raw} exist. Data will be appended. This could be unwanted!"
            logger.warning(msg)

    else:
        msg = f"{args.f_raw} does not exist. Creating new file"
        logger.info(msg)

    try:
        build_raw(
            f_daq=args.f_daq,
            f_raw=args.f_raw,
            jagged=args.jagged,
            ticks_value_ns=args.tick_value,
            trigger_offset=args.shift,
            verbose=args.verbose,
            logger=logger,
        )
    except Exception as e:
        msg = f"An error occurred while building raw tier: {e}"
        logger.error(msg)


if __name__ == "__main__":
    main()
