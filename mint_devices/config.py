#!/usr/bin/python

DATA_OUT_PATH = "/mnt/HDD2/data/P-ONE-ONE"  # Will need to change
PLOT_DIR=''

HEMISPHERE_ID_1 = None  # asks for name at runtime if None
HEMISPHERE_ID_2 = None 
FLASH_UBASE=False

INTERPOSER_TEST=True
ADC_TEST=True

DATABASE_ACTIVE = True
PMT_SUB=False #Flag to select if only a subset of PMT should be turn on
PMT_LIST_H1=np.array([1,2,3,4,5,6,7,8])
PMT_LIST_H2=np.array([1,2,3,4,5,6,7,8])

DARK_RATE=True
HS_SCAN=True

FLASHER_TEST=False
ACCOUSTIC_TEST=False

MIST_TEST=False
CAMERA_TEST=False

PCAL=False #Maybe we don't need it 


SJB=False

MIDAS=False

#Pulser tuning
TUNE=True
TUNE_START=3
TUNE_STEP=1
