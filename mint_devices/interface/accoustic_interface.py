
#This file generates the functions to ensure that the gain of the accoustic receiver is being correctly set. 

for frequency in [10e3, 30e3, 50e3]:
  for gain in [1,2,4]:
    Set_Gain(gain)
    Set_Frequency(frequency)
    Record_data()

for gain in [1...4096]:
  Set_Gain(gain)
  Set_RTM_Sweep()
  Record_data()
  
  
  
  
  
  