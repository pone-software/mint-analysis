import subprocess
import numpy as np
import paramiko
import sys
sys.path.append('/home/vgousy/Documents/Desktop/p-om_production_database')

from p_om_db.omcu_measurements.db_Measurement import db_Measurement
import linregress
def interposer_funtionality_test(inter):
    """
    Testing  the interposer functionality 
    
    """

    #Voltage must be 0-1V
    MIST_biasV = np.array([30,20,10,5])
    SiPM_biasV = np.array([30,20,10,5])
    threshV1 = np.array([0.8,0.5,0.1,0.05])

    #Checking MIST threshold and HV
    inter.enable_muon_tracker_sipm_power(True)
    measure_th_1=[]
    measure_th_2=[]
    measure_th_3=[]
    measure_th_4=[]
    measure_hv=[]
    measure_hv=[]
    measure_hv=[]
    measure_hv=[]
    measure_flasher_hv=[]
    measure_sipm_hv=[]
    
    for j in range(0,4):
        measure_th_1.append(inter.set_mist_threshold(channel=0,voltage=threshV1[j]))
        measure_th_2.append(inter.set_mist_threshold(channel=1,voltage=threshV1[j]))
        measure_th_3.append(inter.set_mist_threshold(channel=2,voltage=threshV1[j]))
        measure_th_4.append(inter.set_mist_threshold(channel=3,voltage=threshV1[j]))
        #measure_flasher_hv.append(i.read_adc_channel(channel=8))
        measure_hv.append(inter.set_muon_tracker_hv(True,SiPM_biasV[j]))
        measure_sipm_hv.append(inter.set_cal_sipm_hv(True,SiPM_biasV[j]))
    print("mist check done")
        #measure_sipm_hv.append(i.read_adc_channel(channel=5))
        #Flasher voltage setting and reading
    inter.enable_5v_power(True)
    inter.enable_flasher_hv(True)
    FLSH_biasW=np.array([5,10])
    FLSH_biasV = np.array([35,20,10,5])
    measure_flasher_0=[]
    measure_flasher_1=[]
    for j in range(0,4):
            measure_flasher_hv=[]
            for k in range(0,2):
                i.set_flasher_state(0,True,FLSH_biasV[j],FLSH_biasW[k]) #Last is pulse width 
                time.sleep(2)
                measure_flasher_hv.append(i.read_adc_channel(channel=8))
                measure_flasher_0.append(measure_flasher_hv[0])
                measure_flasher_1.append(measure_flasher_hv[1])
                #Linear fit to the voltage set    
    r0=stats.linregress(FLSH_biasV,measure_flasher_0)  
    r1=stats.linregress(FLSH_biasV,measure_flasher_1)     
    print("Flasher check done")
    ubase_uid=[]
    for k in range(0,8):
        print(inter.ubase[int(k)].uid)
        ubase_uid.append(inter.ubase[int(k)].uid)

    print("uBase communication check done")
    #Acceptance                                                                                                                                                                                                                                                                                                                                                                           
    threshold=0.1
    if np.any((measure_th_1-threshV1)>threshold) or np.any((measure_th_2-threshV1)>threshold) or np.any((measure_th_3-threshV1)>threshold) or np.any((measure_th_4-threshV1)>threshold):
        print("Rejected interposer")
    else:
        print("Passed mist checking")
    if np.any((measure_sipm_hv-SiPM_biasV)>threshold):
        print("Rejected interposer")
    else:
        print("Passed Sipm check")
    if np.any((measure_hv-SiPM_biasV)>threshold):
        print("Rejected interposer")
    else:
        print("Passed HV muon tracker check")

    if 1-r0.rvalue>0.1 or 1-r1.rvalue>0.1 :
        print("Issue with the flasher voltage setting")
    else:
        print("Passed flasher check")
def correspondance(PMT_num):
     if PMT_num==1:
          return 6
     elif PMT_num==2:
          return 2
     elif PMT_num==3:
          return 1
     elif PMT_num==4:
          return 5
     elif PMT_num==5:
          return 4#4                                                                                                                                                                                                                                                                                                                                                                       
     elif PMT_num==6:
          return 7
     elif PMT_num==7:
          return 3
     elif PMT_num==8:
          return 0
     else:
        print("PMT number is out of range")    

def flash_ubase(inter):
    """
    Function to flash the firmware of the uBase using a single interposer
    """
    inter.set_uart_reflash(True) #Set the 3.3V pin 
    inter.set_ubase_power(False)  #Power cycle the uBase
    inter.set_ubase_power(True)
    result = subprocess.run(['ls', '-l'], capture_output=True, text=True)
    print("Return code:", result.returncode)
    print("Output:\n", result.stdout)
    print("Error:\n", result.stderr)
    hostname = "raspberrypi.local"  # Or the Pi's IP address
    username = "pone"                 # Or your actual Pi username
    password = "cascadia"          # Replace with your password
    script_path = "~/Desktop/flash.sh"  # Path to your script on the Pi
    # Set up SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname, username=username, password=password)
        stdin, stdout, stderr = client.exec_command(f"bash {script_path}")

        print("Output:")
        print(stdout.read().decode())

        print("Errors (if any):")
        print(stderr.read().decode())

    finally:
        client.close()

def turn_on_hv(inter,hemisphere_id=None,hv=False):
    """
    Function to turn on the HV of the different hemisphere at the correct voltage 
    """
    if hv==True:
        for j in range(0,8):
            inter.ubase[j].quick_scan(85)
    else:
        measurement_db = db_Measurement()
        for i in range(1,9):
            PMT_name, _id = measurement_db.get_PMT_from_position(hemisphere_id, str(i), "Devices")
            print("Finding the right voltage for each",PMT_name)
            nominal_HV = measurement_db.get_PMT_data(PMT_name, "Nominal voltage", "Hamamatsu", "Measurements_Pmt")
            print("Nominal voltage found in database is ", nominal_HV)
            inter.ubase[correspondance(i)].quick_scan(nominal_HV)

def turn_off_hv(inter):
    """
    Function to turn off on the HV of the different hemisphere at the correct voltage"
    """
    for j in range(0,8):
            inter.ubase[j].quick_scan(10)



