#After having initialize the Cliend
from perseus.client import PerseusClient
from perseus.control.interposer import Interposer
from perseus.control.ubase import uBase
import interface.interposer_interface as inter 
import interface.adc_interface as adc
import time
import numpy as np
import config

c = PerseusClient("192.168.127.111:12346")

c.attach("interpi", 'interposer.Interposer', gpiodev="gpiochip0", gpio_reset=1, 
gpio_A0=6, gpio_A1=12, gpio_A2=13, spidev="/dev/spidev0.0", 
serialdev="/dev/serial0") #Connect to the interposer, this imply that a server is running on port 123456
print("Initializing interposer")
for attempt in range(3):
        try:
                c.interpi.initialize()
                print("Initialization successful.")
                break 
        Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt == 1:
            print("All attempts failed. Raising exception.")
            raise  # Raise the exception on final failure
if config.INTERPOSER_TEST: 
        inter.funtionality_test(c.interpi)
if config.FLASH_UBASE==True:
        int_test.flash_ubase(c)
print("Initializing Eval kit")
module = ModuleClient("192.168.127.110")
if config.ADC_TEST==True:
        adc.functionality_test(module,init=True)
if PMT_SUB==False:
        inter.turn_on_hv(c,hemisphere_id=config.HEMISPHERE_ID_1,db=config.DATABASE_ACTIVE)
        inter.turn_on_hv(c,hemisphere_id=config.HEMISPHERE_ID_2,db=config.DATABASE_ACTIVE)
else :
      inter.turn_on_hv_subset(c,PMT_list=config.PMT_LIST_H1,hemisphere_id=config.HEMISPHERE_ID_1,db=config.DATABASE_ACTIVE)
      inter.turn_on_hv_subset(c,PMT_list=config.PMT_LIST_H2,hemisphere_id=config.HEMISPHERE_ID_2,db=config.DATABASE_ACTIVE)
        

if config.DARK_RATE:
        pmt.get_dark_rate()
elif config.HS_SCAN:
        pmt.get_charge_distribution()
elif config.FLASHER_TEST:
        flasher.test_functionality()
elif config.ACCOUSTIC_TEST:
        accoustic.test_functionality()





