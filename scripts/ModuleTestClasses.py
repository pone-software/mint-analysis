"""
File to contain test classes for MINT hardware. Designed to interface with
python sequencer in MIDAS
"""

from time import sleep
import numpy as np
import paramiko

from perseus.client import ModuleClient
from perseus.control import timestamp_dict


class SSHAgent:
    def __init__(self, hostname, username, password):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.client = None

    def __del__(self):
        if self.client:
            self.client.close()

    def connect_to_client(self):
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.client.connect(
                self.hostname, username=self.username, password=self.password
            )
        except Exception as e:
            raise Exception(
                f"Failed to connect to ssh client: {self.username}@{self.hostname} with error: {e}"
            )

    def close_client(self):
        if self.client:
            self.client.close()


class Mainboard:
    def __init__(self, module_name, module_ip):
        self.module = ModuleClient(module_ip)
        self.module_name = module_name


class InterposerTesting:
    """
    Class for interposer test scripts. All functions should enable and configure
    hardware only as required for their functionality. All functions should
    return the interposer to a null state before returning.
    """

    def __init__(self, module, interposer_id, ssh_client, sequencer):
        """

        in:
            - module: module instance from Mainboard class
            - interposer_id: XXX this is switching
            - ssh_client: ssh connection to pi or mainboard
        """
        self.module = module
        self.interposer = module.interposer[interposer_id]
        self.ssh_client = ssh_client
        self.seq = sequencer

        # Check to make sure this has loaded properly
        if hasattr(self.interposer, "load_error"):
            raise Exception(f"Interposer has load error: {self.interposer.load_error}")

    def test_mist(self, threshold_acceptance, hv_acceptance):
        """
        Checks threshold and MIST SiPM high voltage capability.

        Stores acceptance within thresholds in
            output["threshold_acceptance_array"]
            output["hv_acceptance_array"]
        """
        mist_voltages = np.array([30, 20, 10, 5])
        threshold_voltages = np.array([0.8, 0.5, 0.1, 0.05])

        # Store multidimensional output in terms of channels and measurements
        output = {
            "readback_thresholds": np.zeros((4, len(threshold_voltage))),
            "measured_thresholds": np.zeros((4, len(threshold_voltages))),
            "readback_mist_hv": np.zeros(len(mist_voltages)),
            "measured_mist_hv": np.zeros(len(mist_voltages)),
        }

        # measure thresholds
        for i, threshold_voltage in enumerate(threshold_voltages):
            # Set thresholds and get readback values
            for channel in range(4):
                output["readback_thresholds"][channel].append(
                    self.interposer.set_mist_threshold(
                        channel=channel, voltage=threshold_voltage
                    )
                )

            # pause briefly for thresholds to set
            sleep(1)

            # Read thresholds as measured by the onboard ADC
            output["measured_thresholds"][0, i] = self.interposer.read_adc_channel(2)
            output["measured_thresholds"][1, i] = self.interposer.read_adc_channel(3)
            output["measured_thresholds"][2, i] = self.interposer.read_adc_channel(1)
            output["measured_thresholds"][3, i] = self.interposer.read_adc_channel(0)

        # Check acceptance. If any measurement minus the expected is greater
        # than the allowed acceptance, set acceptance to False
        output["threshold_acceptance_array"] = (
            abs(output["measured_voltage"] - threshold_voltages[:, np.newaxis])
            <= threshold_acceptance
        )

        # Enable -3V3 voltage to MIST SiPMs.
        self.interposer.enable_muon_tracker_sipm_power(True)
        # Enable HV to MIST SiPMs. Should this be tested with the backup
        # generator as well?
        self.interposer.enable_flasher_hv(True)

        # Get the current state, make sure these are set
        for i in range(5):
            sleep(1)
            state = self.interposer.get_state()
            if state["neg_3v3_state"] and state["mist_sipm_hv_state"]:
                break
            if i == 4:
                raise Exception("Could not enable MIST power")

        # Set voltage, readback values, then measure
        for i, mist_voltage in enumerate(mist_voltages):
            # Set and get readback
            output["readback_mist_hv"][i] = self.interposer.set_muon_tracker_hv(
                True, mist_voltage
            )
            # pause and meditate
            sleep(2)
            # measure high voltage
            output["measured_mist_hv"][i] = self.interposer.read_adc_channel(4)

        # Check if measurements are within acceptance of the set voltages
        output["hv_acceptance_array"] = (
            abs(output["readback_mist_hv"] - mist_voltages) <= hv_acceptance
        )

        # Turn off all the things we turned on
        self.interposer.set_muon_tracker_hv(False, 0)
        self.interposer.enable_flasher_hv(False)
        self.interposer.enable_muon_tracker_sipm_power(False)

        # check acceptance of HV
        return timestamp_dict(output)

    def test_flasher_power(self, acceptance, use_backup=False):
        """
        Function to test flasher HV and pulse widths

        Stores acceptance within thresholds in
            output["hv_pre_filter_acceptance"]
            output["hv_post_filter_acceptance"]
            output["hv_f1_acceptance"]
        """

        pulse_widths_ns = np.array([5, 10])
        bias_voltages = np.array([35, 20, 10, 5])

        output = {
            "measured_flasher_pre_filter_hv": np.zeros(
                (len(bias_voltages), len(bias_voltages))
            ),
            "measured_flasher_post_filter_hv": np.zeros(
                (len(bias_voltages), len(bias_voltages))
            ),
            "measured_flasher_hv_f1": np.zeros(
                (len(bias_voltages), len(bias_voltages))
            ),
        }

        # Enable 5V, flasher HV
        self.interposer.enable_5v_power(True)
        self.interposer.enable_flasher_hv(True, use_backup)

        # Check state, make sure 5V is set
        for i in range(5):
            sleep(1)
            state = self.interposer.get_state()
            if abs(state["5v_voltage"] - 5) < 0.5:
                break
            if i == 4:
                raise Exception("Could not enable 5V power")

        # Cycle through voltages and widths. (Do we need channels too?)
        # Use first channel for now, just measuring a global voltage.
        # Does the width actually change the main HV?
        for j, bias_voltage in enumerate(bias_voltages):
            for i, pulse_width in enumerate(pulse_widths_ns):
                # Set state
                self.interposer.set_flasher_state(0, True, bias_voltage, pulse_width)
                # Meditate some more
                sleep(2)
                # Measure
                output["measured_flasher_pre_filter_hv"][i, j] = (
                    self.interposer.read_adc_channel(6)
                )
                output["measured_flasher_post_filter_hv"][i, j] = (
                    self.interposer.read_adc_channel(7)
                )
                output["measured_flasher_hv_f1"][i, j] = (
                    self.interposer.read_adc_channel(8)
                )

        output["hv_pre_filter_acceptance"] = (
            abs(output["measured_flasher_pre_filter_hv"] - bias_voltages[:, np.newaxis])
            <= acceptance
        )
        output["hv_post_filter_acceptance"] = (
            abs(
                output["measured_flasher_post_filter_hv"] - bias_voltages[:, np.newaxis]
            )
            <= acceptance
        )
        output["hv_f1_acceptance"] = (
            abs(output["measured_flasher_hv_f1"] - bias_voltages[:, np.newaxis])
            <= acceptance
        )

        # Turn off flasher and voltages
        self.interposer.set_flasher_state(0, False, 0, 0)
        self.interposer.enable_flasher_hv(False, use_backup)
        self.interposer.enable_5v_power(False)

        return timestamp_dict(output)

    def test_microbase_presence(self, selected_microbases):
        """
        Check each microbase, make sure that no load error is present and that
        we can read back the uid.

        Stores acceptance in

        """
        # Deconvolve indices. Remove this when perseus is updated
        ubase_indices = [6, 2, 1, 5, 4, 7, 3, 0]

        output = {"ubase_uids": np.empty(8)}

        # Go through microbases and check if we can read their IDs
        for microbase in selected_microbases:
            # Check if there's a load error
            if hasattr(self.interposer.ubase[ubase_indices[microbase]], "load_error"):
                output["ubase_uids"][microbase] = False
                continue
            # if not, store uid
            output["ubase_uids"][microbase] = self.interposer.ubase[
                ubase_indices[microbase]
            ].uid

        return timestamp_dict(output)

    def flash_microbases(self, selected_microbases, script_path):
        """
        Function to flash the selected microbases
        """
        error_status = 1
        for microbase in selected_microbases:
            # Set up forwarding to the correct microbase
            self.interposer.select_ubase(microbase)
            # Power cycle the microbase to put it into flashing mode
            self.interposer.set_ubase_power(False)
            self.interposer.set_ubase_power(True)
            # Meditate
            sleep(2)
            # Use the ssh client to run the bash script that reflashes the microbases
            try:
                _, stdout, stderr = self.client.exec_command(f"bash {script_path}")
                self.seq.msg(
                    f"Microbase {microbase} stdout: {stdout.read().decode()}, stderr: {stderr.read().decode()}"
                )
            except Exception as e:
                self.seq.msg(
                    f"Failed reflash on microbase {microbase} with exception: {e}, stdout: {stdout.read().decode()}, and stderr: {stderr.read().decode()}"
                )
                error_status = -1

        return error_status

    def turn_on_hv(self, hv=False):
        """
        Function to set the HV of a module either to the optimal value measured in the lab

        """
