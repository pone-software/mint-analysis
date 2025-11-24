"""
This module contains a class for reading data in the ENG format.
"""

from __future__ import annotations

allowed_lengths = {
    "ChannelThresholds": 17,
    "CoincidenceWindow": 2,
    "BufferAllTriggers": 2,
    "TimeSync": 5,
    "Trigger": 4,
}
waveform_field_length_min = 3


class Readout:
    def __init__(self, channels=0, timestamp=0, since_previous=0):
        self.channels = channels
        self.timestamp = timestamp
        self.since_previous = since_previous
        self.waveforms = {}

    def valid(self):
        return self.channels and len(self.waveforms) == self.nChannels()

    def __bool__(self):
        return self.valid()

    def has_channel(self, c: int):
        return self.channels & (15 - c)

    def nChannels(self):
        c = 0
        v = self.channels
        while v:
            c += v & 1
            v >>= 1
        return c

    def get_waveform(self, c: int):
        if c not in self.waveforms:
            msg = f"Waveform {c} not found"
            raise RuntimeError(msg)
        return self.waveforms[c]


class EngFormatReader:

    def __init__(self, in_stream):
        self._instream = in_stream
        self.line_number = 0
        self.eof = False
        self.channel_thresholds = [4096] * 16
        self.coincidence_window = 0
        self.buffer_all_triggers = False
        self.time_sync_words = [0] * 4

    @staticmethod
    def clean_line(line):
        line = line.strip()
        idx = line.find("#")
        if idx != -1:
            line = line[0:idx]
        return line

    def decode_channel_thresholds(self, fields):
        if len(fields) != allowed_lengths["ChannelThresholds"]:
            msg = f"Wrong number of fields for ChannelThresholds record on line {self.line_number}"
            raise ValueError(msg)
        try:
            self.channel_thresholds = [
                int(t) for t in fields[1 : allowed_lengths["ChannelThresholds"]]
            ]
        except ValueError as e:
            raise ValueError(
                e.args[0] + f" for channel threshold value on line {self.line_number}"
            ) from e

    def decode_coincidence_window(self, fields):
        if len(fields) != allowed_lengths["CoincidenceWindow"]:
            msg = f"Wrong number of fields for CoincidenceWindow record on line {self.line_number}"
            raise ValueError(msg)
        try:
            self.coincidence_window = int(fields[1])
        except ValueError as e:
            raise ValueError(
                e.args[0] + f" for coincidence window value on line {self.line_number}"
            ) from e

    def decode_buffer_all_triggers(self, fields):
        if len(fields) != allowed_lengths["BufferAllTriggers"]:
            msg = f"Wrong number of fields for BufferAllTriggers record on line {self.line_number}"
            raise ValueError(msg)
        if fields[1].lower() == "true" or fields[1] == "1":
            self.buffer_all_triggers = True
        elif fields[1].lower() == "false" or fields[1] == "0":
            self.buffer_all_triggers = False
        else:
            msg = f"Invalid boolean value '{fields[1]}' for"
            msg += f"BufferAllTriggers record on line {self.line_number}"
            raise ValueError(msg)

    def decode_time_sync(self, fields):
        if len(fields) != allowed_lengths["TimeSync"]:
            msg = f"Wrong number of fields for TimeSync record on line {self.line_number}"
            raise ValueError(msg)
        try:
            self.time_sync_words = [
                int(w, base=16) for w in fields[1 : allowed_lengths["TimeSync"]]
            ]
        except ValueError as e:
            raise ValueError(e.args[0] + f" for time sync word on line {self.line_number}") from e

    def decode_trigger(self, ro, fields):
        if len(fields) != allowed_lengths["Trigger"]:
            msg = f"Wrong number of fields for Trigger record on line {self.line_number}"
            raise ValueError(msg)
        try:
            ro.channels = int(fields[1], base=16)
        except Exception:
            msg = f"Malformed Trigger record on line {self.line_number}:"
            msg += "Invalid channel mask"
            raise ValueError(msg) from None
        waveforms_to_read = ro.nChannels()
        # 				print("waveforms_to_read is now",waveforms_to_read)
        try:
            ro.timestamp = int(fields[2])
        except Exception:
            msg = f"Malformed Trigger record on line {self.line_number}: Invalid timestamp"
            raise ValueError(msg) from None
        try:
            ro.since_previous = int(fields[3])
        except Exception:
            msg = f"Malformed Trigger record on line {self.line_number}:"
            msg += "Invalid number of samples since previous trigger"
            raise ValueError(msg) from None
        return waveforms_to_read

    def decode_waveform(self, ro, fields):
        if len(fields) < waveform_field_length_min:
            msg = f"Too few fields for Waveform record on line {self.line_number}"
            raise ValueError(msg)

        try:
            channel = int(fields[1])
        except Exception:
            msg = f"Malformed Waveform record on line {self.line_number}:"
            msg += "Invalid channel number"
            raise ValueError(msg) from None
        try:
            n_samples = int(fields[2])
        except Exception:
            msg = f"Malformed Waveform record on line {self.line_number}:"
            msg += "Invalid number of waveform samples"
            raise ValueError(msg) from None
        if len(fields) != 3 + n_samples:
            msg = "Wrong number of fields for Waveform"
            msg += "record on line {self.line_number}"
            raise ValueError(msg)
        try:
            ro.waveforms[channel] = [int(i) for i in fields[3:]]
        except Exception:
            msg = f"Malformed Waveform record on line {self.line_number}:"
            msg += "invalid or missing waveform sample"
            raise ValueError(msg) from None

    def read_next_trigger(self):
        waveforms_to_read = 0
        ro = Readout()
        while True:
            self.line_number += 1
            line = self._instream.readline()
            if not line:
                self.eof = True
                return None
            # 			print("Raw line:", line)
            line = EngFormatReader.clean_line(line)
            # 			print("Cleaned line:", line)
            if len(line) == 0 or len(line.split()) < 1:
                continue
            fields = line.split()
            tag = fields[0]

            if waveforms_to_read and tag != "Waveform":
                msg = f"Unexpected record of type '{tag}' on line {self.line_number}; "
                msg += f"{waveforms_to_read} more waveform(s) were expected"
                raise ValueError(msg)
            if tag == "ChannelThresholds":
                self.decode_channel_thresholds(fields)
            elif tag == "CoincidenceWindow":
                self.decode_coincidence_window(fields)
            elif tag == "BufferAllTriggers":
                self.decode_buffer_all_triggers(fields)
            elif tag == "TimeSync":
                self.decode_time_sync(fields)
            elif tag == "Trigger":
                waveforms_to_read = self.decode_trigger(ro, fields)
            elif tag == "Waveform":
                self.decode_waveform(ro, fields)
                waveforms_to_read -= 1
                # 				print("waveforms_to_read is now",waveforms_to_read)
                if waveforms_to_read == 0:
                    break
            else:
                msg = f"Unrecognized record tag '{tag}' on line {self.line_number}"
                raise ValueError(msg)
        return ro
