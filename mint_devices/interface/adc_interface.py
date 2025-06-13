import socket
import time
import numpy as np
import subprocess
import paramiko
#from eng_reader import EngFormatReader
import os
import matplotlib.pyplot as plt
import shutil
import re
import sys
sys.path.insert(1, "/home/mint-daq/p-om_production_database")
from p_om_db.omcu_measurements.db_Measurement import db_Measurement

#import sys
#sys.path.insert(1, "/home/ecp_pi/Desktop/remote_readout/munich_pmt_calibration_system/omcu")
HOST = '0.0.0.0'# Listen on all available interfaces                                                                                                  PORT = 12347                  # Port for listening for the main computer                                                                                            
# Eval kit details
hostname = '192.168.127.110'
port = 22  # Default SSH port
username = 'root'
password = 'pone'
command = " eng_readout --host '192.168.127.100' -p 9000 -s 24 -q 6 -b 4 -t 5"
executable_path="/home/mint-daq/fastDAQ/build/./eng_collector"
# The command you want to run                                                                                                                                                                                                                                                                                                                        
ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, port=port, username=username, password=password)
def functionality_test(module,ini=False):
    """
    Simple function checking that the ADC can be aligned correctly
    """
    status=module.adc.rx_status()
    for idx, lane in enumerate(status.lanes):
        if lane.valid == 0:
            raise RuntimeError(f"Error: rx_status on lane {idx} is not 0 (found {lane.rx_status})")
        if not status.global_aligned:
            raise RuntimeError("Error: global alignment failed (global_aligned is False)")
    if ini:
         module.adc.init() #Initialize the ADC  
def get_baseline(module):
    """
    Get the baseline of every channel in correct order
    """
    return module.trigger.channel_means()[::-1]#Return in reverse order
def get_rms(module):
    """
    Get the standard error of every channel in correct order
    """
    #np.flip(arr)
    #list.reverse()
    return module.trigger.channel_rms()[::-1]
def get_threshold(channel,hemisphere_id):
    """Get the threshold from the database"""
    PMT_name, _id = measurement_db.get_PMT_from_position(hemisphere_id, str(channel), "Devices")
    threshold=measurement_db.get_PMT_data(self,pmt_id,'Threshold measurement','TUM','Measurements_Pmt')
    if threshold==0 or threshold<2:
        return 3
    else :
        return threshold
def update_threshold_all(path,baseline,threshold):
    """
    Update 16 threshold based on the input 
    """
    coincidence_line = "CoincidenceWindow 0"
    buffer_line = "BufferAllTriggers false"
    threshold_values=list(np.array(baseline)+np.array(threshold))
    filename = path+"config.txt"
    with open(filename, 'w') as f:
        f.write('#Example configuration file\nChannelThresholds '+' '.join(str(int(x)) for x in threshold_values))
        f.write('\nCoincidenceWindow 0'+'\nBufferAllTriggers false')
def update_threshold_channel(path,channel,baseline,threshold):
    """
    Only update the threshold of a single channel
    """
    coincidence_line = "CoincidenceWindow 0"
    buffer_line = "BufferAllTriggers false"
    filename = path+'pmt_'+str(channel)+'config.txt'
    threshold_values=np.ones(16)*4095
    threshold_values[channel]=baseline+threshold
    with open(filename, 'w') as f:
        f.write('#Example configuration file\nChannelThresholds '+' '.join(str(int(x)) for x in list(threshold_values)))
        f.write('\nCoincidenceWindow 0'+'\nBufferAllTriggers false')
    
def get_waveform_all(module,hemisphere_1_id,hemisphere_2_id,path,database_active=False,dark_rate=1,nr_waveforms=10000):
    """
    Record waveform on all of the ADC channel
    """
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port=port, username=username, password=password)
    baseline=get_baseline(module)
    threshold_h1=[]
    threshold_h2=[]
    if database_active==False:
        threshold=list(np.ones(16)*5)
    else:
        for i in range(1,17):
            if i>8:
                threshold_h2.append(get_threshold(i-8,hemisphere_id_2))
            else:
                threshold_h1.append(get_threshold(i,hemisphere_id_1))
        threshold=threshold_h1+threshold_h2
    #Update the config file
    print(threshold)
    update_threshold_all(path,baseline,threshold) 
    path='/home/vgousy/Documents/Desktop/bash_scripts/'    
    if dark_rate:
        arguments=['-c',path+'config.txt','-o','dark_rate_all'+'.txt']
        command_board="eng_readout --host 192.168.127.100 -b 20 -q 25 -s 40 -t 10"
    else:
        arguments=['-c',path+"config_flash_trigger.txt",'-o','charge_calibration_all'+'.txt']
        command_board="eng_readout --host 192.168.127.100 -c" +str(2**(16))+"-b 5 -q 15 -s 40 -m "+str(nr_waveforms) # Getting charge distribution

    process=subprocess.Popen([executable_path]+arguments, stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)                                                                                                                                                                                                                                                                                  
    stdin, stdout, stderr = ssh.exec_command(command_board)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')                                                                                                                                                                                                                                                                                                             
    print("Output:")
    print(output)
    ssh.close() 
    
def get_waveform(module,channel,hemisphere_id,path,threshold=3,dk=False,nr_waveforms=5000,occ=False,database_active=False):
    """
    Record waveform on a single channel specified
    """
    
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port=port, username=username, password=password)
    
    baseline=get_baseline(module)[channel]
    if database_active==False:
        threshold=5
    else:
        threshold=get_threshold(channel,hemisphere_id)
    if 0<channel<17:
        if dk==True:
            update_threshold_channel(path,channel,baseline,threshold)
            arguments=['-c',path+'pmt_'+str(channel)+'config.txt','-o',path+'dark_rate_channel'+'_'+str(channel)+'.txt']
            command_board="eng_readout --host 192.168.127.100 -b 20 -q 25 -s 40 -t 10"
        elif occ==True:
            arguments=['-c','pmt_ref.conf','-o',path+'occ_channel'+'_'+str(channel)+'_'+str(voltage)+'.txt']                                                                                                                                                                                                                                                                                                                                                                      
            command_board="./eng_readout --host 192.168.127.100 -c "+str(2**(channel+1)+2**(channel_ref))+" -s 40 -q 15 -b 5 -m "+str(nr_waveforms)
        else:
            arguments=['-c','pmt_ref.conf','-o',path+'charge_channel'+'_'+str(channel)+'_'+str(voltage)+'.txt']
            command_board="./eng_readout --host 192.168.127.100 -c "+str(2**(channel+1)+2**(channel_ref))+" -s 40 -q 1 -b 0 -m "+str(nr_waveforms)
        
    
    process=subprocess.Popen([executable_path]+arguments, stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)                                                              
                                                              
                                                                                                                                                                                                                                                                                        
    stdin, stdout, stderr = ssh.exec_command(command_board)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')      
    ssh.close()
    # Print the results                                                                                                                                                                                                                                
    print("Output:")
    print(output)
#def get_occ(module):
#    """
#    Collect the intensity of the photons 
#    """
#    #
    
    