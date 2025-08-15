import os
import sys
import time
import pyvisa
import datetime
import subprocess
import numpy as np
import h5py as hp
from optparse import OptionParser

from perseus.client import ModuleClient
from RTM3004 import RTM3004


class Acoustics:
    """A class to record data on the mainboard over perseus with the audio TLV320ADC6140"""

    ###############################################################################
    # PARAMETERS
    ###############################################################################
    # mainboard Connection

    def __init__(self, **kwargs):
        """
        Required input arguments:
            - mainboard_ip
            - output_dir
            - tag
            - debug

        kwargs used in case extra arguments are folded into this function.
        """
        self.time_str = datetime.datetime.now().isoformat("_")[:-7].replace(":", "")

        # Configure from midas sequencer in kwargs
        # Mainboard IP "11.102.1.1"
        self.debug = kwargs["debug"]
        self.output_dir = kwargs["output_dir"]
        self.tag = kwargs["tag"]
        self.output_dir = os.path.join(self.output_dir, self.tag)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Hardcoded?
        self.t_record = None
        self.process_response = None
        self.default_gain = 1
        self.adc_status = False

        # osci_address "11.102.0.106"
        self.rtm = RTM3004(device=kwargs["osci_address"])
        self.module = ModuleClient(kwargs["mainboard_ip"])
        self.seq = kwargs["seq"]

        self.acoustic_out = None
        self.acoustic_err = None

    def __del__(self):
        self.close_collection_port()

    def set_record_time(self, time):
        self.t_record = time

    def return_record_time(self):
        return self.t_record

    def startRecording(self, fname, ag=[0, 0, 0, 0]):
        output_file = os.path.join(self.output_dir, fname)
        if self.t_record == None:
            raise ValueError("Run time was not assigned to t_record.")
        cmd = ""  # Some command for setting the listen port and reading into binary then converting. Converting can be done in post before analysis
        time.sleep(3)
        return output_file

    def osci_on(self, amp_V):
        self.rtm.toggleWaveform(status="ON")
        self.rtm.setWaveFunction("SIN")
        # set in volts
        self.rtm.setWaveVoltage(amp=amp_V)
        self.rtm.setWaveVoltOffset(0)
        self.rtm.toggleWaveformBurst(status="OFF")

    def osci_off(self):
        self.rtm.toggleWaveform(status="OFF")
        self.rtm.stopAquisition()

    def enable(self, gain_1=None, gain_2=None):
        if not gain_1:
            gain_1 = self.default_gain
        if not gain_2:
            gain_2 = self.default_gain
        self.module.interposer_a.SetAcousticState(True, gain_1, gain_2)
        self.module.interposer_b.SetAcousticState(True, gain_1, gain_2)

    def disable(self):
        self.module.interposer_a.SetAcousticState(
            False, self.default_gain, self.default_gain
        )
        self.module.interposer_b.SetAcousticState(
            False, self.default_gain, self.default_gain
        )

    def adc_enable(self):
        # if self.adc_status:
        # return
        self.module.acoustic.initialize()
        time.sleep(0.1)
        self.module.acoustic.set_active()
        time.sleep(0.1)
        self.module.acoustic.set_output_config(0, 2, 0, 0, 0, 0)
        time.sleep(0.1)
        self.module.acoustic.set_communication(6, 5)
        time.sleep(0.1)
        self.module.acoustic.set_clock_source()
        time.sleep(0.1)

        self.module.acoustic.set_channel_slot(1, 0, 0)
        self.module.acoustic.set_channel_slot(2, 0, 1)
        time.sleep(0.1)

        self.module.acoustic.set_gain(1, 1)
        self.module.acoustic.set_gain(2, 1)
        time.sleep(0.1)

        self.module.acoustic.set_input_config(0, 1, 1, 0, 1, 1)
        self.module.acoustic.set_input_config(0, 1, 1, 0, 1, 2)
        time.sleep(0.1)

        self.module.acoustic.set_dynamic_range_enhancer(0, 11, False)
        time.sleep(0.1)

        self.module.acoustic.set_input_enable(1, True)
        self.module.acoustic.set_input_enable(2, True)
        time.sleep(0.1)

        self.module.acoustic.set_output_enable(1, True)
        self.module.acoustic.set_output_enable(2, True)
        time.sleep(0.1)

        self.module.acoustic.get_status()

        self.module.acoustic.set_adc_power(False, True, 0, 0)

        # self.adc_status = True

    def sweepRTM(
        self,
        start_f_hz=10e3,
        end_f_hz=50e3,
        sweep_t_s=5,
        run_t_s=15,
        record=True,
        amp_V=0.01,
        gain_per_stage=None,
    ):
        if not gain_per_stage:
            gain_per_stage = self.default_gain

        self.seq.msg(f"Runing acoustic sweep from {start_f_hz} to {end_f_hz} Hz.")
        self.rtm.reset()
        offset = 3
        self.set_record_time(run_t_s + offset)
        self.enable(gain_per_stage, gain_per_stage)

        if record:
            self.start_recording(f"sweep_gain{gain_per_stage**2}")

        self.osci_on(amp_V=amp_V)
        self.rtm.setWaveVoltFrequency(freq=1e3)

        self.rtm.setSweepTime(sweep_t_s)
        self.rtm.setStartFreqSweep(start_f_hz)
        self.rtm.setEndFreqSweep(end_f_hz)
        self.rtm.setSweepType(style="LIN")
        self.rtm.toggleSweep(toggle="ON")

        time.sleep(run_t_s)
        self.rtm.toggleSweep(toggle="OFF")
        self.osci_off()
        time.sleep(offset)

        # self.wait_for_raspi() # << Something equivalent for the mainboard?

    def burstSweepRTM(
        self,
        start_f_hz=10e3,
        end_f_hz=50e3,
        steps=10,
        cycles=20,
        per_step_time=10,
        record=True,
        amp_V=0.01,
        gain_per_stage=None,
    ):
        if not gain_per_stage:
            gain_per_stage = self.default_gain

        freq_range = np.linspace(start_f_hz, end_f_hz, steps)
        total_time = per_step_time * steps
        # reset scope
        self.rtm.reset()

        self.seq.msg(f"Runing acoustic burst sweep from {start_f_hz} to {end_f_hz} Hz.")
        self.rtm.reset()
        self.set_record_time(total_time + 5)

        self.osci_on(amp_V=amp_V)
        self.rtm.setWaveVoltFrequency(freq=1e3)
        self.rtm.toggleSweep(toggle="OFF")
        self.enable(gain_per_stage, gain_per_stage)

        if record:
            self.startRecording(f"burst_sweep_gain{gain_per_stage**2}")

        self.rtm.setWaveformBurstCount(cycles=cycles)
        self.rtm.setWaveformBurstIdle(0.5)
        self.rtm.toggleWaveformBurst()

        for freq in freq_range:
            self.rtm.setWaveVoltFrequency(freq=freq)
            time.sleep(per_step_time)

        self.rtm.toggleWaveformBurst(status="OFF")
        self.osci_off()

        # self.wait_for_raspi()

        return freq_range

    def sanityCheckRTM(self, record=True, gains_per_stage=[1, 2, 4], amp_V=0.01):
        # reset scope
        self.seq.msg(f"Runing acoustic sanity check at 10, 30, and 50 kHz.")
        self.rtm.reset()
        time.sleep(1)
        self.set_record_time(40)
        if record:
            self.startRecording(
                f"sanity_check_gains-{(gains_per_stage[0])**2}-{(gains_per_stage[1])**2}-{(gains_per_stage[2])**2}"
            )
        time.sleep(0.01)
        self.osci_on(amp_V)
        self.enable()
        self.rtm.setWaveVoltFrequency(freq=1e3)

        freq_vals = [10e3, 30e3, 50e3]

        for freq in freq_vals:
            self.rtm.setWaveVoltFrequency(freq=freq)
            for gain in gains_per_stage:
                self.enable(gain, gain)
                time.sleep(3)

        self.osci_off()
        # self.wait_for_raspi()

        return freq_vals

    def singleFreqRun(self, record=True, freq=10e3, gain_per_stage=1, amp_V=0.01):
        # reset scope
        self.seq.msg(f"Running single frequency check at {freq}.")
        self.rtm.reset()
        time.sleep(1)
        self.set_record_time(30)
        if record:
            self.startRecording(f"single_frequency{freq}_gain{gain_per_stage**2}")
        time.sleep(0.01)
        self.osci_on(amp_V)

        self.enable(gain_per_stage, gain_per_stage)
        self.rtm.setWaveVoltFrequency(freq=freq)
        time.sleep(30)

        self.osci_off()
        # self.wait_for_raspi()

    def backgroundRun(self, record=True, amp_V=0.01):
        # reset scope
        self.seq.msg(f"Runing acoustic acoustic background check.")
        self.rtm.reset()
        time.sleep(1)
        gainlist_per_stage = [1, 2, 4, 8, 16, 32]
        gain_count = len(gainlist_per_stage)
        delta_t = 10
        self.set_record_time(delta_t * gain_count + 5)
        gain_title = ""
        for i, gain in enumerate(gainlist_per_stage):
            gain_title += f"-{gain**2}"
        if record:
            self.startRecording("background_gains" + gain_title)
        time.sleep(0.01)
        self.osci_on(amp_V)

        for gain in gainlist_per_stage:
            self.enable(gain, gain)
            self.rtm.setWaveVoltFrequency(freq=1e3)
            time.sleep(delta_t)

        self.osci_off()
        # self.wait_for_raspi()

    def open_collection_port(
        self,
        port,
        output_filename="./output/acoustic_out.bin",
        err_filename="./output/acoustic_err.txt",
    ):
        """
        Opens a netcat port to receive binary data from the acoustic stream off
        of a mainboard.

        Will return immediately and allow the process to run on a new thread.
        """

        self.acoustic_out = open(output_filename, "wb")
        self.acoustic_err = open(err_filename, "w")
        self.listen_proc = subprocess.Popen(
            f"nc -l {port}", stdout=self.acoustic_out, stderr=self.acoustic_err
        )

    def close_collection_port(self):
        """
        Closes collection port and open files.
        """
        if self.listen_proc.poll() is None:
            self.listen_proc.kill()
        if self.acoustic_out is not None and not self.acoustic_out.closed:
            self.acoustic_out.close()
        if self.acoustic_err is not None and not self.acoustic_err.closed:
            self.acoustic_err.close()
