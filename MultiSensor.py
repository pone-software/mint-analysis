from __future__ import annotations
import serial
import time
from perseus.control import timestamp_dict
#from . import timestamp_dict
#from perseus.client import PerseusClient
#from perseus.client import ModuleClient

__all__ = ["MultiSensor"]

class MultiSensor:
    """
    Class for working with the multi sensor of the MINT
    """
    print("Starting....")
    

    __tuber_object__ = True
    __tuber_exclude__ = ["sensor", "device"]

    def __init__(self, serial_port: str, verbose: bool = False):
        """
        Initialize serial connection to DLP-TH1C sensor
        """
        self.sensor = serial.Serial(serial_port, 115200, timeout=1)
        self.sensor.read_all() 
        time.sleep(0.5)

        self.verbose = verbose

    def initialize():
        return timestamp_dict(self.get_all())
    def __del__(self):
        self.sensor.close()
        del self.sensor

    def __query(self, msg : bytearray) -> str:
        e=self.sensor.write(msg)
        time.sleep(1)
        val = self.sensor.read_all().decode("unicode_escape")
        time.sleep(1)
        e = self.sensor.read_all()
        return val

    
    def get_temperature(self) -> float:
        """
        Return the current temperature in C
        """
        return float(self.__query(b"t\n").split("= ")[1].split('°C')[0])
    
    def get_pressure(self) -> float:
        """
        Return the current pressure in hPa
        """
        return float(self.__query(b"p\n").split("= ")[1].split('hpa')[0])
            
    def get_humidity(self) -> float:
        """
        Return the current humidity in %
        """
        return float(self.__query(b"h\n").split("= ")[1].split('%')[0])
    
    def get_light_level(self) -> int:
        """
        Return the current light level between 0 to 255
        """
        return int(self.__query(b"l\n").split(": ")[1])
    
    def get_broadband_sound_level(self) -> float:
        """
        Returns the broadband ambient sound level
        """

        numbers=[]
        for word in self.__query(b"f\n").split():
            cleaned = ''.join(c for c in word if c.isdigit() or c == '.')
            if cleaned.replace('.', '', 1).isdigit():
                numbers.append(float(cleaned))
            if len(numbers) == 2:
                break
        return numbers
    
    def get_sound(self) -> tuple:
        """
        Returns the fundamental (peakamplitude) frequency of ambient sound and
        five lower-amplitude peaks. (Note that
        lower-amplitude peaks can be above or below the fundamental.) 
        """
        return(self.__query(b"f\n"))
    
    def get_tilt(self) -> tuple:
        return self.__query(b"a\n")
    
    def get_x_vibration(self) -> tuple:
        """
        Returns the fundamental (peakamplitude) frequency of vibration and
        five lower-amplitude peaks. Along X axis. (Note that
        lower-amplitude peaks can be above or below the fundamental.) 
        """
        return self.__query(b"x\n")
    
    def get_y_vibration(self) -> tuple:
        """
        Returns the fundamental (peakamplitude) frequency of vibration and
        five lower-amplitude peaks. Along Z axis. (Note that
        lower-amplitude peaks can be above or below the fundamental.) 
        """
        return self.__query(b"v\n")
    
    def get_z_vibration(self) -> tuple:
        """
        Returns the fundamental (peakamplitude) frequency of vibration and
        five lower-amplitude peaks. Along Z axis. (Note that
        lower-amplitude peaks can be above or below the fundamental.) 
        """
        return self.__query(b"w\n")
    

    def get_all(self) -> dict:
        return{"temperature": self.get_temperature(),
               "pressure": self.get_pressure(),
               "humidity": self.get_humidity(),
               "light level": self.get_light_level(),
               "broadband sound": self.get_broadband_sound_level(),
               "sound": self.get_sound(),
               "tilt": self.get_tilt(),
               "x vibrations": self.get_x_vibration(),
               "y vibrations": self.get_y_vibration(),
               "z vibrations": self.get_z_vibration(),

        }
    #def get_x_vibration(self, sensitivity="2G"):
     #   """
      #  Set the sensitivity of the vabration sensor.
       # Allowed values for sensitivity are: 2G, 4G, 8G, 16G
        #"""
        #raise NotImplementedError
