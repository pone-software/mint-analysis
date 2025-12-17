"""
Takes selective data from the ANA tier and uploads it to the mongoDB
mongoDB structure has the shape:
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import yaml
from andromeda.testing.devices.production_database import ProductionDatabase
from pint import UnitRegistry
from pint.errors import UndefinedUnitError

from .utils import get_physics_object, quantity_to_dict, setup_logging

ROOT_DIR = Path("/home/pkrause/software/mint-analysis/debug_out")

# binary channel mask to upload
# 0x0001 is PMT 1 only
# 0x0002 is PMT 2 only
# 0x0003 is PMT 1,2
# 0x0004 is PMT 3 only


NumberOrList = float | List[float]
StrOrList = str | List[str]


@dataclass
class Mapping:
    parameters: list[str]
    func: str


@dataclass
class PMTInfo:
    v10_in_volt: NumberOrList
    di10_in_mA: NumberOrList
    frac: NumberOrList


@dataclass
class EnvironmentInfo:
    temperature_in_celsius: NumberOrList
    pressure_in_hpa: NumberOrList
    humidity_in_percent: NumberOrList
    measurement_duration_in_s: NumberOrList


@dataclass
class SoftwareInfo:
    framework: str
    pe_reconstruction: str  # e.g. "NNLS" | "Integral"
    sftp_path: str
    run_tags: StrOrList


@dataclass
class MeasurementResult:
    measurement_type: str
    measurement_location: str
    devices_used: Any  # TODO: Loosely typed right now, can be tightened with mongoDB object?

    result: NumberOrList
    result_unc: NumberOrList
    units: StrOrList
    insert_time: str = field(init=False)

    pmt_info: PMTInfo
    env_info: EnvironmentInfo
    software_info: SoftwareInfo

    mapping: Mapping | None = None

    def __post_init__(self):
        # ---- Type consistency: list or scalar ----
        is_list_result = isinstance(self.result, list)
        is_list_unc = isinstance(self.result_unc, list)
        is_list_units = isinstance(self.units, list)

        if len({is_list_result, is_list_unc, is_list_units}) != 1:
            msg = "result, result_unc, and units must all be either lists or scalars"
            raise ValueError(msg)

        # ---- Mapping existence rule ----
        if self.mapping is not None and not is_list_result:
            msg = "mapping may only be provided when result, result_unc, and units are lists"
            raise ValueError()

        # ---- Length consistency (only for list case) ----
        if is_list_result:
            lengths = {
                len(self.result),
                len(self.result_unc),
                len(self.units),
            }

            if self.mapping is not None:
                lengths.add(len(self.mapping.parameters))

            if len(lengths) != 1:
                msg = "result, result_unc, units, and mapping.parameters must all have same length"
                raise ValueError(msg)

        self.insert_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def get_values(
    obj: Any,
) -> tuple[NumberOrList, NumberOrList, StrOrList, Mapping | None, list | None]:
    if hasattr(obj, "magnitude") and hasattr(obj, "units"):
        ret = quantity_to_dict(obj)
        if "err" not in ret:
            return ret["val"], 0, ret["unit"], None, None
        return ret["val"], ret["err"], ret["unit"], None, None
    # else we have multiple results (e.g. gain fit)
    if isinstance(obj, dict):
        ret = quantity_to_dict(obj)
        vals = []
        errs = []
        units = []
        params = []
        for k, v in ret.items():
            if not isinstance(v, dict):
                continue
            vals.append(v["val"])
            errs.append(v.get("err", 0))
            units.append(v["unit"])
            params.append(k)

        return (
            vals,
            errs,
            units,
            Mapping(params, ret.get("func", "No mapping function provided")),
            ret.get("used_keys", []),
        )

    msg = f"{type(obj)} not supported"
    raise NotImplementedError(msg)


def get_pmt_info(channel: int, aux_dict: dict, keys: str | list) -> PMTInfo:

    if isinstance(keys, str):
        v10 = aux_dict[keys][channel]["v10"]
        di10 = aux_dict[keys][channel]["di10"]
        frac = aux_dict[keys][channel]["frac"]
        return PMTInfo(v10.to("V").m, di10.to("mA").m, frac.m)

    if isinstance(keys, list):
        v10 = []
        di10 = []
        frac = []
        for k in keys:
            v10.append(aux_dict[k][channel]["v10"].to("V").m)
            di10.append(aux_dict[k][channel]["di10"].to("mA").m)
            frac.append(aux_dict[k][channel]["frac"].m)
        return PMTInfo(v10, di10, frac)

    msg = f"{type(keys)} not supported"
    raise NotImplementedError(msg)


def get_env_info(aux_dict: dict, keys: str | list) -> EnvironmentInfo:
    """
    Return average env info across runs
    """
    if isinstance(keys, str):
        pth_start = aux_dict[keys]["pth_start"]
        pth_end = aux_dict[keys]["pth_end"]
        t = (pth_end["temperature"].to("degC").m + pth_start["temperature"].to("degC").m) / 2
        p = (pth_end["pressure"].to("hPa").m + pth_start["pressure"].to("hPa").m) / 2
        h = (pth_end["humidity"].to("percent").m + pth_start["humidity"].to("percent").m) / 2
        m = aux_dict[keys]["runtime"].to("seconds").m
        return EnvironmentInfo(t, p, h, m)

    if isinstance(keys, list):
        t, p, h, m = [], [], [], []
        for key in keys:
            pth_start = aux_dict[key]["pth_start"]
            pth_end = aux_dict[key]["pth_end"]
            t.append(
                (pth_end["temperature"].to("degC").m + pth_start["temperature"].to("degC").m) / 2
            )
            p.append((pth_end["pressure"].to("hPa").m + pth_start["pressure"].to("hPa").m) / 2)
            h.append(
                (pth_end["humidity"].to("percent").m + pth_start["humidity"].to("percent").m) / 2
            )
            m.append(aux_dict[key]["runtime"].to("seconds").m)

        return EnvironmentInfo(t, p, h, m)

    msg = f"{type(keys)} not supported"
    raise NotImplementedError(msg)


def auto_int(x):
    """
    Custom type converter for argparse that attempts to convert a string
    to an integer, supporting both decimal and hexadecimal (prefixed with '0x').
    """
    try:
        # Attempt to convert as a decimal integer
        return int(x)
    except ValueError:
        try:
            # If decimal conversion fails, attempt to convert as hexadecimal
            # The base 0 in int(x, 0) handles '0x' prefix automatically
            return int(x, 0)
        except ValueError as e:
            msg = f"Invalid integer or hexadecimal value: '{x}'"
            raise argparse.ArgumentTypeError(msg) from e


def main():
    logger = setup_logging(level=logging.INFO)
    ureg = UnitRegistry()

    parser = argparse.ArgumentParser(description="Upload analysis results to mongoDB.")
    parser.add_argument("-x", "--f_aux", help="Path to aux file", required=True)
    parser.add_argument("-a", "--f_ana", help="Path to ana file", required=True)
    parser.add_argument(
        "-d", "--dry", action="store_true", help="Perform a dry run only (i.e do not upload to DB)"
    )
    parser.add_argument(
        "-c",
        "--ch_mask",
        type=auto_int,
        help="Channel mask (can be decimal or hexadecimal, e.g., 10 or 0xA)",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--measurement",
        choices=["dcr", "linear_gain", "snr"],
        help="Measurement to upload",
        required=True,
    )
    parser.add_argument(
        "-k",
        "--key",
        default=None,
        help="key to upload (will be ignored for linear_gain measurement)",
    )
    parser.add_argument(
        "-r", "--reco", default="NNLS", help="Reconstruction algorithm used in DSP"
    )

    parser.add_argument("--mongo_user", default="mint", help="Specify user for mongoDB login")
    parser.add_argument(
        "--mongo_keyring",
        default=None,
        help="Specify keyring for mongoDB login (will prompt password if None)",
    )
    parser.add_argument(
        "--ssh_keyring",
        default=None,
        help="Specify keyring for ssh tunnel login  (will prompt password if None)",
    )
    parser.add_argument("--ssh_user", default="mint", help="Specify user for ssh tunnel login")

    args = parser.parse_args()
    result_yaml = args.f_ana
    aux_file = args.f_aux
    measurement = args.measurement
    reco = args.reco
    ch_mask = args.ch_mask
    tag = args.key
    db_opts = {
        "mongo_user": args.mongo_user,
        "use_tunnel": True,
        "ssh_keyring_service": args.ssh_keyring,
        "ssh_user": args.ssh_user,
        "mongo_keyring_service": args.mongo_keyring,
        "logger": logger.info,
    }

    # Build SFTP directory
    sftp_pmt_dir = "PMT"
    sftp_base_dir = Path("/mint/mint-data/")
    p = Path(aux_file)
    try:
        index = p.parts.index(sftp_pmt_dir)
        sftp_root_dir = Path(p.parts[index - 1])
        sftp_dir = str(sftp_base_dir / sftp_root_dir / sftp_pmt_dir)
    except ValueError:
        msg = f"'{sftp_pmt_dir}' not found in the aux path."
        logger.error(msg)

    # Load measurement data
    try:
        with open(result_yaml) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        msg = f"Failed to load file: {e}"
        logger.error(msg)
        raise

    # 'calibration_constants' in aux file define NNLS
    if "calibration_constants" in data:
        calib_data = get_physics_object(data["calibration_constants"], ureg)
        nnls_coloumb_factor = (
            calib_data.get("vadc")
            * calib_data.get("upsampling_ratio")
            * calib_data.get("sampling_time")
            * calib_data.get("renormalization_factor")
        ) / calib_data.get("adc_impedance")
        ureg.define(f"NNLS = {nnls_coloumb_factor.to('coulomb').magnitude} * coulomb")
        ureg.define(f"ADC = {calib_data.get('upsampling_ratio').magnitude}*NNLS")

    # if no calibration values passed, try to load data without NNLS parameters
    # if NNLS is in the tagged measurement, raise error.
    try:
        data = get_physics_object(data, ureg)
    except UndefinedUnitError as e:
        if "NNLS" in str(e):
            msg = "Unit NNLS not defined. Trying to load reduced data set"
            logger.warning(msg)
            data = {measurement: get_physics_object(data[measurement], ureg)}
        else:
            raise e

    # Load aux file
    try:
        with open(aux_file) as f:
            aux = yaml.safe_load(f)
        aux = get_physics_object(aux, ureg)
    except Exception as e:
        msg = f"Failed to load file: {e}"
        logger.error(msg)
        raise

    # check measurement sanity
    if measurement not in data:
        msg = f"{measurement} measurement not in result file"
        logger.error(msg)
        return

    tags_aux = list(aux.keys())
    if measurement == "linear_gain":
        if tag is not None:
            logger.warning(
                "Tags are defined redundandly. Using tags defined in linear gain results"
            )
        tag = data[measurement].get("used_keys")
        if len(tag) < 3:
            logger.warning("Less than three measurements found. Linear fit result will be bogus")

        if not set(tag).issubset(set(tags_aux)):
            msg = f"Tag {tag} not found in the aux file"
            logger.error(msg)
            raise ValueError

    else:
        # Check key sanity
        tags_data = list(data[measurement].keys())
        if tag not in tags_aux:
            msg = f"Tag {tag} not found in the aux file"
            logger.error(msg)
            raise ValueError
        if tag not in tags_data:
            msg = f"Tag {tag} not found in the result file"
            logger.error(msg)
            raise ValueError

    while ch_mask:
        ch = (ch_mask & -ch_mask).bit_length() - 1
        pmt_no = ch + 1

        # collect value(s)
        vals, errs, units, mapping, _keys = get_values(
            data[measurement][pmt_no]
            if measurement == "linear_gain"
            else data[measurement][tag][pmt_no]
        )
        logger.info("collected measurement results")

        # collect pmt info
        pmt_info = get_pmt_info(pmt_no, aux, tag)
        logger.info("collected PMT info")

        # get environment info
        env_info = get_env_info(aux, tag)
        logger.info("collected environment info")

        sw_info = SoftwareInfo(
            framework="mint-xyz", pe_reconstruction=reco, sftp_path=sftp_dir, run_tags=tag
        )
        logger.info("collected software info")

        # get used devices
        # Get settings from DB
        # TODO replace with module level logic
        with ProductionDatabase(**db_opts) as db:
            overview = db.get_hemisphere_overview(str(sftp_root_dir), "tum")
            pmt_obj = {
                "device_type": "pmt-unit",
                "uid": overview.get("pmt-unit").get(str(pmt_no)).get("uid"),
                "_id": overview.get("pmt-unit").get(str(pmt_no)).get("_id"),
            }

        # Build measurement result
        res = MeasurementResult(
            measurement_type=measurement,
            measurement_location="MINT",
            devices_used=pmt_obj,
            result=vals,
            result_unc=errs,
            units=units,
            mapping=mapping,
            pmt_info=pmt_info,
            env_info=env_info,
            software_info=sw_info,
        )
        logger.info("Build data object")

        # Upload to database
        if args.dry:
            msg = f"The following dict would be uploaded to the DB: {asdict(res)}"
        else:
            with ProductionDatabase(**db_opts) as db:
                db.client["mint"]["Measurements_Pmt"].insert_one(asdict(res))
            msg = f"Uploaded document {asdict(res)} to mint/Measurements_Pmt"

        logger.info(msg)
        ch_mask &= ch_mask - 1  # clear lowest set bit


# --------------------------
# CLI entrypoint
# --------------------------
if __name__ == "__main__":
    main()
