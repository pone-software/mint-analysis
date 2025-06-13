from ../devices import K6482





def flasher_functionality(inter,channel,voltage,tolerance=1000e-6):
    """
    Function to check if the flasher is indeed turning on or not using a picoamp 
    """
    pico=K6482() #Initialize picoamp
    dark_current=np.mean(pico.readCurrent(10))
#Turn on the Flasher with interposer 
    inter.enable_5v_power(True)
    inter.enable_flasher_hv(True)
    inter.set_flasher_state(channel,True,voltage,10) #Channel number, voltage(v), pulse width (nsec)
    current=np.mean(pico.readCurrent("10"))
    #Will need some specific mainboard commands
    if (current-dark_current)>tolerance:
        print("Flasher tested")
    else:
        print("Issue with the flasher")
        
        
        
"""
    
[Establsh communication to MB]->Done in the main script
[Establish communication to IPB]->Also done in the main script
---
Set FG to 1kHz
For CH in [0 - 7]
  For V in [5, 24]
    Set flasher HV for ${CH} to ${HV}
    Enable flasher HV for ${CH}
    Read flasher HV for ${CH}
    Enable FG output
    Enable flasher ${CH}
    Read PMT signals (relative to FG trigger)
    

"""


