import midas
import midas.client
import sys
import re
import csv
from OscilloscopeClass import Oscilloscope
import numpy as np
from ScopeVisualizer import ScopePlot
#client = midas.client.MidasClient("frontendclient")
#client.odb_set("/Equipment/Oscilloscope/Variables/Connected",1)
#except Exception as e:
 #   print("Cannot initialize the Oscilloscope")
  #  print(f"Error: {e}")
   # print(client.odb_get("/Equipment/Oscilloscope/Variables/Connected"))
    #client.odb_set("/Equipment/Oscilloscope/Variables/Connected",0)
    #sys.exit(1)

sys.path.insert(0, "/home/mint-daq/midas/python")
import midas.frontend
import midas.event
import os

#This is the working one
class OscilloscopeFront(midas.frontend.EquipmentBase):
    """
    We define an "equipment" for each logically distinct task that this frontend
    performs. For example, you may have one equipment for reading data from a
    device and sending it to a midas buffer, and another equipment that updates
    summary statistics every 10s.
    
    Each equipment class you define should inherit from 
    `midas.frontend.EquipmentBase`, and should define a `readout_func` function.
    If you're creating a "polled" equipment (rather than a periodic one), you
    should also define a `poll_func` function in addition to `readout_func`.
    """
    def __init__(self, client):
        print("Initializing....")
        # The name of our equipment. This name will be used on the midas status
        # page, and our info will appear in /Equipment/MyPeriodicEquipment in
        # the ODB.
        equip_name = "Oscilloscope"
        
        response = os.system("ping 11.102.0.106")
        print(response,"response")
        if response==0:
            print("Scope is up!")
        else:
            print("Scope is down")
        # Define the "common" settings of a frontend. These will appear in
        # /Equipment/MyPeriodicEquipment/Common. The values you set here are
        # only used the very first time this frontend/equipment runs; after 
        # that the ODB settings are used.
        default_common = midas.frontend.InitialEquipmentCommon()
        default_common.equip_type = midas.EQ_PERIODIC
        default_common.buffer_name = "SYSTEM"
        default_common.trigger_mask = 0
        default_common.event_id = 1
        default_common.period_ms = 1000
        default_common.read_when = midas.RO_RUNNING
        default_common.log_history = 1
        print("after all the commons set")
        
        
        # You MUST call midas.frontend.EquipmentBase.__init__ in your equipment's __init__ method!
        midas.frontend.EquipmentBase.__init__(self, client, equip_name, default_common)
        print("after the midas frontend line")
        # You can set the status of the equipment (appears in the midas status page)
        self.set_status("Initialized")
    


    def readout_func(self):
        print("Readout function called")        

class MyFrontend(midas.frontend.FrontendBase):
    """
    A frontend contains a collection of equipment.
    You can access self.client to access the ODB etc (see `midas.client.MidasClient`).
    """
    def __init__(self):
        
        print("init, front end class")
        # You must call __init__ from the base class.
        midas.frontend.FrontendBase.__init__(self, "Oscilloscope")
        print("registering odbwatch")
        param = '-n' if os.sys.platform.lower()=='win32' else '-c'
        #hostname="11.102.0.106"
        hostname = self.client.odb_get("/Equipment/Oscilloscope/Variables/DeviceIP")
        print(hostname, "hostnametest")
        response = os.system(f"ping {param} 1 {hostname}")
        if response==0:
            print("Oscilloscope is Connected!")
        else:
            print("Oscilloscope is Not Connected!")
        #Watches for update in ODB Waveshape
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/Shape", self.handle_shape_change)
        #self.client.odb_watch("/Equipment/Oscilloscope/Variables/test", self.handle_test)
        #Watches for update in ODB Frequency
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/Frequency", self.handle_frequency_change)
        #Watches for update in ODB Voltage
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/Voltage", self.handle_voltage_change)
        #Watches for update in "Checking", which, while true, prints the current values directly from the Oscilloscope
        #After this happens, it should set itself to fasle
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/Checking", self.handle_check_change)
        #Watches for update in the Sync, which while true pushes the values from the ODB to the scope. This should happen automatically.
        #After this happens, it should set itself to false
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/Sync", self.handle_sync)
        #Watches for update in ODB autoscaling, which, while true, will trigger the Oscilloscope's autoscale feature
        #It will then set itself to false
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/Autoscaling", self.handle_autoscaling)
        #Checks for updates in the gain in the ODB
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/Gain", self.handle_gain_change)
        #Checks for updates in the FunctionGenerator ODB, then toggles the FunctionGenerator accordingly
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/FunctionGenerator", self.handle_FG_toggle)
        #Checks the Channel on the ODB
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/Channel",self.watch_channel)
        #Checks for updates on the trigger slope status
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/TriggerSlope",self.Update_Trigger_Slope)
        #Checks for update in the datapath variable in the ODB. It will create a file of the oscilloscope data in
        #The local directory
        self.client.odb_watch("/Equipment/Oscilloscope/Variables/DataPath",self.handle_get_data)
        print(self.client.odb_get("Equipment/Oscilloscope/Variables/Frequency"))
        #This doesn't actually seem to be working, but it was designed to set defaults when the
        #Script runs. Not sure why it's working at the moment        
        self.client.odb_set("/Equipment/FridgeSensor/Variables/Checking",0)
        self.client.odb_set("/Equipment/FridgeSensor/Variables/Channel",1)
        self.client.odb_set("/Equipment/FridgeSensor/Variables/Sync",0)


        self.client.odb_watch("/Equipment/Oscilloscope/Variables/Heartbeat",self.handle_heartbeat)

        print("registered odbwatch")
    #Toggles the Function Generator. key1 is self and value is 0 or 1, corresponding with off or on
    #def handle_test(key1,key2,path,value):
        #print("test successful")

    def handle_heartbeat(key1,key2,path,value):
        if key1.client.odb_get("Equipment/Oscilloscope/Variables/Heartbeat")==1:
            param = '-n' if os.sys.platform.lower()=='win32' else '-c'
            hostname = key1.client.odb_get("/Equipment/Oscilloscope/Variables/DeviceIP")
            print(hostname, "hostnametest")
            response = os.system(f"ping {param} 1 {hostname}")
            if response==0:
                print("Oscilloscope is Connected!")
                key1.client.odb_set("Equipment/Oscilloscope/Variables/Connected",1)
            else:
                print("Oscilloscope is Not Connected!")
                key1.client.odb_set("Equipment/Oscilloscope/Variables/Connected",0)
            key1.client.odb_set("Equipment/Oscilloscope/Variables/Heartbeat",0)

    def handle_FG_toggle(key1,key2,path,value):
        print("_____________________")
        if value==1:
            print("Turning on")
            scope.output_on()
        if value==0:
            print("Turning off")
            scope.output_off()
    #Handles the sync, which updates the scope with the ODB values, then stops. This should probably never ben called
    def handle_sync(key1,key2,path,value):
        print("please don't spin infinitely")
        if key1.client.odb_get("Equipment/Oscilloscope/Variables/Sync")==1:
            print("_____________________")
            _channel=key1.client.odb_get("Equipment/Oscilloscope/Variables/Channel")
            scope.wave_type(key1.client.odb_get("Equipment/Oscilloscope/Variables/Shape"))
            scope.write_frequency(key1.client.odb_get("Equipment/Oscilloscope/Variables/Frequency"))
            scope.write_voltage(key1.client.odb_get("Equipment/Oscilloscope/Variables/Voltage"))
            scope.SetGain(_channel, key1.client.odb_get("Equipment/Oscilloscope/Variables/Gain"))
            print("Synced!")
            key1.client.odb_set("Equipment/Oscilloscope/Variables/Sync",0)

    #Trigger's the scope's autoscaling function
    def handle_autoscaling(key1,key2,path,value):
        if key1.client.odb_get("Equipment/Oscilloscope/Variables/Autoscaling")==1:
            print("_____________________")
            scope.AutoScale()
            key1.client.odb_set("Equipment/Oscilloscope/Variables/Autoscaling",0)
    #Just prints the ODB channel update. Nothing fancy
    def watch_channel(key1,key2,path,value):
        print("_____________________")
        print(f"Updated Channel to {value}")
    #Grabs the data and writes it to a .csv locally
    def handle_get_data(key1, key2, path, value):
        print("_____________________")
        channel = key1.client.odb_get("Equipment/Oscilloscope/Variables/Channel")
        filename = key1.client.odb_get("Equipment/Oscilloscope/Variables/DataPath")

        try:
            data_array = scope.GetData(channel)  # now returns a float32 NumPy array

            # Save clean float32 binary data
            with open(filename, "wb") as f:
                f.write(data_array)

            print("Saved first 10 float32 samples:", data_array[:10])
            print(f"Saved {len(data_array)} float32 samples to {filename}")

            # Plot it
            ScopePlot.plot_downsampled(filename, 100)

        except Exception as e:
            print("Error in handle_get_data:", e)

    #Changes the trigger slope status
    def Update_Trigger_Slope(key1, key2, path, value):
        print("_____________________")
        print(value,"value")
        if (value=="Rising"):
            scope.trigger_slope_update("POS")
        if (value=="Falling"):
            scope.trigger_slope_update("NEG")
        if (value=="Either"):
            scope.trigger_slope_update("EITH")
    #Changes the shape of the function generator's wave
    def handle_shape_change(key1, key2, path, value):
        print("_____________________")
        scope.wave_type(value)
    #Changes the frequency of the function generator's wave
    def handle_frequency_change(key1, key2, path, value):
        print("_____________________")
        scope.write_frequency(value)
    #Changes the voltage of the function generator's wave
    def handle_voltage_change(key1, key2, path, value):
        print("_____________________")
        scope.write_voltage(value)
    #Changes the gain of the function generator's wave. For some reason this is changed for like a second then changed back to 1.
    #This is probably fine since we basically always want the gain to be 1 anyway
    def handle_gain_change(key1,key2,path,value):
        print("_____________________")
        scope.SetGain(1,value)
    #This reads the values directly from the osciloscope (except the channel, which is read from the ODB)
    #And prints it
    #This is done this way since Midas seems to have the easiest time seeing updates in ODB
    #values, so when the "get parameters" button is pressed, it updates some bool to 1
    #Then, this function happens, and since the bool is 1, the if statement triggers
    #Then, the function sets that bool to 0.
    #The function technically triggers again, but the bool is now 0, so the if statement doesn't trigger
    #Ending the "loop" after just one instance, until the button is pressed again
    #Weird line, I know. There's probably a better way to do this, but it works
    def handle_check_change(key1, key2, path, value):
        if key1.client.odb_get("/Equipment/Oscilloscope/Variables/Checking")==1:
            print("_____________________")
            _freq=scope.read_frequency()
            _volt=scope.read_voltage()
            _shape=scope.GetWave()
            _channel=key1.client.odb_get("/Equipment/Oscilloscope/Variables/Channel")
            _gain=scope.GetGain(_channel)
            
            _toggled=key1.client.odb_get("/Equipment/Oscilloscope/Variables/FunctionGenerator")
            if _toggled==1:
                print("The Function Generator is On")
            if _toggled==0:
                print("The Function Generator is Off")
            print("Channel:",_channel)
            print("Frequency:",_freq)
            print("Voltage:",_volt)
            print("Wave Shape:",_shape)
            print("Gain (should be 1):",_gain)
            key1.client.odb_set("/Equipment/Oscilloscope/Variables/Checking",0)
    
    def begin_of_run(self, run_number):
        """
        This function will be called at the beginning of the run.
        You don't have to define it, but you probably should.
        You can access individual equipment classes through the `self.equipment`
        dict if needed.
        """
        print("Beginning of Run")
        self.set_all_equipment_status("Running", "greenLight")
        self.client.msg("Frontend has seen start of run number %d" % run_number)
        return midas.status_codes["SUCCESS"]
        
    def end_of_run(self, run_number):
        self.set_all_equipment_status("Finished", "greenLight")
        self.client.msg("Frontend has seen end of run number %d" % run_number)
        return midas.status_codes["SUCCESS"]
    
    def frontend_exit(self):
        OscilloscopeFront.client.odb_set("Equipment/Oscilloscope/Variables/Connected",0)
        print(OscilloscopeFront.client.odb_get("Equipment/Oscilloscope/Variables/Connected"))
        """
        Most people won't need to define this function, but you can use
        it for final cleanup if needed.
        """
        print("Goodbye from user code!")
        
if __name__ == "__main__":
    print("Frontend file found")
    try:
        scope = Oscilloscope()  # move here!
        with MyFrontend() as OscilloscopeFront:
            OscilloscopeFront.run()
    except KeyboardInterrupt:
        print("Received Ctrl-C, shutting down...")
        try:
            OscilloscopeFront.stop()
        except NameError:
            pass  # OscilloscopeFront was never created
    except Exception as e:
        print("Exception occurred:", e)
    finally:
        print("Cleanup complete. Exiting.")

