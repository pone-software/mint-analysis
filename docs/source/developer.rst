Developers Guide
================

This package is designed to process data from the MINT deployment. 
It uses the LEGEND Data Objects package  <https://legend-pydataobj.readthedocs.io> to handle/store 
the data and DSPEED <https://dspeed.readthedocs.io> to process the data. 

All new processors at the moment should be added to DSPEED for processing. The config 
file under configs can then be used to specify the processing chain to use.