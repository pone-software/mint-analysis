from RsInstrument import RsInstrument
import sys
import midas.client
import time
import numpy as np
class Oscilloscope:
    def __init__(self):
        client = midas.client.MidasClient("pytest")
        scopeIP = client.odb_get("/Equipment/Oscilloscope/Variables/DeviceIP")
        self.connect_to_scope(scopeIP,client)
        #scopeIP = '11.102.0.106'
    def connect_to_scope(self,scopeIP,client):
        try:
            self.instr = RsInstrument(
                f'TCPIP::{scopeIP}::inst0::INSTR',
                id_query=True,
                reset=False,
                options="SelectVisa='rs'"
            )
            print("test")
            client.odb_set("/Equipment/Oscilloscope/Variables/Connected",1)
        except BaseException as e:
            print(f"Couldn't connect to the Oscilloscope at {scopeIP}")
            print(f"Error: {e}")
            client.odb_set("/Equipment/Oscilloscope/Variables/Connected",0)
            time.sleep(10)
            scopeIP = client.odb_get("/Equipment/Oscilloscope/Variables/DeviceIP")
            self.connect_to_scope(scopeIP,client)
            #sys.exit(1)
        #print(self.instr.query("*IDN?"))
        #print(self.instr.query("*OPT?"))

    def read_frequency(self):
        return self.instr.query("WGEN:FREQ?")

    def read_voltage(self):
        return self.instr.query("WGEN:VOLT?")

    def write_frequency(self, frequency):
        self.instr.write(f"WGEN:FREQ {frequency}")
        print(f"Frequency updated to {frequency}!")

    def write_voltage(self, voltage):
        self.instr.write(f"WGEN:VOLT {voltage}")
        print(f"Voltage updated to {voltage}!")

    def output_on(self):
        self.instr.write("WGEN:OUTP ON")

    def output_off(self):
        self.instr.write("WGEN:OUTP OFF")

    def wave_type(self, wavetype):
        #Types are DC, SINuisoid, SQUare, PULSe, TRIangle, RAMP, SINC, ARBitrary, and EXPonential
        self.instr.write(f"WGEN:FUNC {wavetype}")
        print(f"Shape changed to {wavetype}!")

    def GetWave(self):
        return self.instr.query(":WGEN:FUNC?")

    def GetData(self,channel):
        data_str = self.instr.query(f":CHANnel{channel}:DATA?")
        data = np.fromstring(data_str, sep=",")
        binary_data=data.astype(np.float32)
        return binary_data

    

    def GetGain(self,channel):
        print(self.instr.query(f"PROBe{channel}:SETup:GAIN?"))
        return self.instr.query(f"PROBe{channel}:SETup:GAIN?")

    def SetGain(self,channel,gain):
        print("Setting Gain:",gain)
        return self.instr.write(f"PROBe{channel}:SETup:GAIN:MANual {gain}")





    def GetFreq(self):
        self.instr.write("AUToscale")
        self.instr.write("MEASurement1:SOURce CHAN1")
        self.instr.write("MEASurement1:TYPE FREQuency")
        frequency_str = self.instr.query("MEASurement1:RESult?")
        print(frequency_str, "in function printing")
        return float(frequency_str)
    
    def AutoScale(self):
        self.instr.write("AUToscale")
        print("Activated the Oscilloscope's Autoscale Function")

    def trigger_slope_update(self, value):
        self.instr.write(f"TRIGger:A:EDGE:SLOPe {value}")
        print(f"Updated the trigger slope to {value}")

    def run_test(self):
        freq = self.read_frequency()
        print(freq, "is the current frequency")

        self.write_frequency("95")
        self.write_voltage("0.45")
        self.output_on()
        self.wave_type("SQU")
        print("About to get frequency")
        #This doesn't work, I don't think there is a way to get the input frequency directly, I'll have to write code to analyze it
        #print(self.GetFreq())
        #print(self.GetData())
        print(self.instr.query("WGEN:FREQ?"))
        print("Wave output enabled on the RTM3004 function generator")

if __name__ == "__main__":
    scope = Oscilloscope()
    scope.run_test()
