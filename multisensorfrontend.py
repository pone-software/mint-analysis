import midas
import sys
import re
from MultiSensor import MultiSensor
sys.path.insert(0, "/home/mint-daq/midas/python")
import midas.frontend
import midas.event
#This is the working one
class FridgeSensor(midas.frontend.EquipmentBase):
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
        equip_name = "FridgeSensor"
        # Define the "common" settings of a frontend. These will appear in
        # /Equipment/MyPeriodicEquipment/Common. The values you set here are
        # only used the very first time this frontend/equipment runs; after 
        # that the ODB settings are used.
        default_common = midas.frontend.InitialEquipmentCommon()
        default_common.equip_type = midas.EQ_PERIODIC
        default_common.buffer_name = "SYSTEM"
        default_common.trigger_mask = 0
        default_common.event_id = 1
        default_common.period_ms = 10000
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
        """
        For a periodic equipment, this function will be called periodically
        (every 10000ms in this case). It should return either a `midas.event.Event`
        or None (if we shouldn't write an event).
        """
        serial_port = "/dev/ttyACM0"
        #serial_port = "/dev/ttyUSB0"  # On Linux or Mac
        #serial_port = "COM7"        # On Windows

        sensor = MultiSensor(serial_port, verbose=True)


        #temperature =  float(self.__query(b"t\n").split("= ")[1].split('°C')[0])
        temperatureValue = sensor.get_temperature()
        pressureValue = sensor.get_pressure()
        humidityValue = sensor.get_humidity()
        lightValue = sensor.get_light_level()
        soundValue = sensor.get_broadband_sound_level()
        xVibrationValue = sensor.get_x_vibration()
        yVibrationValue = sensor.get_y_vibration()
        zVibrationValue = sensor.get_z_vibration()
        print(temperatureValue)
        
        # In this example, we just make a simple event with one bank.
        event = midas.event.Event()
        
        # Create a bank (called "MINT") which in this case will store 8 ints.
        # data can be a list, a tuple or a numpy array.
        #data = [temperature,pressure,humidity]
        temperature = [temperatureValue]
        pressure = [pressureValue]
        humidity = [humidityValue]
        light = [lightValue]
        sound=soundValue
        #sound = re.findall(r'\d+',soundValue)
        #sound = sound[:2]
        #sound = list(map(int,sound))
        xVibration=re.findall(r'\d+',xVibrationValue)
        xVibration=xVibration[:2]
        xVibration=list(map(int,xVibration))
        yVibration=re.findall(r'\d+',yVibrationValue)
        yVibration=yVibration[:2]
        yVibration=list(map(int,yVibration))
        zVibration=re.findall(r'\d+',zVibrationValue)
        zVibration=zVibration[:2]
        zVibration=list(map(int,zVibration))
        self.client.odb_set("Equipment/FridgeSensor/Variables/Temperature",temperatureValue)
        self.client.odb_set("Equipment/FridgeSensor/Variables/Pressure",pressureValue)
        self.client.odb_set("Equipment/FridgeSensor/Variables/Humidity",humidityValue)
        self.client.odb_set("Equipment/FridgeSensor/Variables/Light",lightValue)
        self.client.odb_set("Equipment/FridgeSensor/Variables/Sound",soundValue)
        self.client.odb_set("Equipment/FridgeSensor/Variables/XVibration",xVibration)
        self.client.odb_set("Equipment/FridgeSensor/Variables/YVibration",yVibration)
        self.client.odb_set("Equipment/FridgeSensor/Variables/ZVibration",zVibration)
        #For sound and vibrations, the first number is the frequency
        #and the second number is the amplitude
        #event.create_bank("TEMP", midas.TID_FLOAT, temperature)
        #event.create_bank("PRES", midas.TID_FLOAT, pressure)
        #event.create_bank("HMDY", midas.TID_FLOAT, humidity)
        #event.create_bank("LITE", midas.TID_INT, light)
        #event.create_bank("SOUN", midas.TID_FLOAT, sound)
        #event.create_bank("XVIB", midas.TID_INT, xVibration)
        #event.create_bank("YVIB", midas.TID_INT, yVibration)
        #event.create_bank("ZVIB", midas.TID_INT, zVibration)
        #event.create_bank("MINT", midas.TID_FLOAT, data)
        
        return event

class MyFrontend(midas.frontend.FrontendBase):
    """
    A frontend contains a collection of equipment.
    You can access self.client to access the ODB etc (see `midas.client.MidasClient`).
    """
    def __init__(self):
        print("init, front end class")
        # You must call __init__ from the base class.
        midas.frontend.FrontendBase.__init__(self, "FridgeSensor")
        self.client.register_jrpc_callback(self.my_rpc_callback, True)
        print("registering odbwatch")
        #self.client.odb_watch("/Equipment/FridgeSensor/Variables/TEMP", self.handle_odb_change)
        equip_name="FridgeSensor"
        path=f"/Equipment/{equip_name}/Common"
        #self.client.odb_set_default(f"{path}/Buffer","SYSTEM")
        #self.client.odb_set_default(f"{path}/Enabled","True")
        #self.client.odb_set_default(f"{path}/Event ID", 1)
        #self.client.odb_set_default(f"{path}/Trigger mask", 0)
        #self.client.odb_set_default(f"{path}/Period", 10000)
        #self.client.odb_set_default(f"{path}/Read on", 1)
        #self.client.odb_set_default(f"{path}/Log history", 1)


        print("registered odbwatch")
        # You can add equipment at any time before you call `run()`, but doing
        # it in __init__() seems logical.
        self.add_equipment(FridgeSensor(self.client))

    def my_rpc_callback(client, cmd, args, max_len):
        print("oh man it worked")
               
    def handle_odb_change(key1, key2, path, value):
        print(f"ODB value changed at {path}: {value}")

    
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
        """
        Most people won't need to define this function, but you can use
        it for final cleanup if needed.
        """
        print("Goodbye from user code!")
        
if __name__ == "__main__":
    print("Frontend file found")
    try:
        with MyFrontend() as FridgeSensor:
            FridgeSensor.run()
    except KeyboardInterrupt:
        print("Received Ctrl-C, shutting down...")
        FridgeSensor.stop()
    finally:
        print("Cleanup complete. Exiting.")

