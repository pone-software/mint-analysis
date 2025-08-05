import time
import pyvisa
import numpy as np
import logging

class RTM3004():
    '''This class creates an instance for the RTM3004 oscilloscope
    as a pyvisa instrument.
    '''
        
    def __init__(self, device='TCPIP::scp-8417-osc.phys.sfu.ca::INSTR'):
        print('Initializing RTM3004 oscilloscope...')
        ResourceManager = pyvisa.ResourceManager()
        self.instrument = ResourceManager.open_resource(device)
        self.instrument.timeout = 60*1000
        self.name = device
        self.connected = True
        self.SimpleMeasurementStatus = False
        self.SimpleSetupStatus = False        
        self.wavevolt = 0
        self.reset()
        print('... done.\n')

    def write(self, message):
        self.instrument.write(message)
        time.sleep(0.01)
        return

    def ask(self, message):
        response = self.instrument.query(message)
        return response

    def identify(self):
        return self.ask('*IDN?')
   
    def reset(self):
        self.write('*RST')

    def wait(self, dt=0.5):
        while True:
            done = self.ask('*OPC?')
            if done == '1\n':
                break
            time.sleep(dt)

    def getTermination(self, channel=1):
        return self.ask('PROB%i:SET:IMP?' %(channel))
    
    def setBandwidth(self, channel=1, bw='FULL'):
        # bandwidth of the channel, MHz
        # available values FULL|B20
        self.write('CHAN%i:BAND %s' %(channel, bw))

    def getBandwidth(self, channel=1):
        return self.ask('CHAN%i:BAND?' %(channel))

    ######################################################
    # HORIZONTAL 
    ######################################################

    def setHorizontalScale(self, div=50e-4):
        # scale of one div in horizontal, seconds [1e-6, ...] s/div
        self.write('TIM:SCAL %.2e' %(div))

    def getHorizontalScale(self):
        return self.ask('TIM:SCAL?')

    def setHorizontalPosition(self, t=0):
        # horizontal position of trigger edge on display
        self.write('TIM:POS %.6f' %(t))

    def getHorizontalPosition(self):
        return self.ask('TIM:POS?')

    ######################################################
    # VERTICAL 
    ######################################################

    def setVerticalScale(self, channel=1, div=5e-4):
        # scale of one div in vertical, Volts, [1e-3, 10]
        self.write('CHAN%i:SCAL %.3f' %(channel, div))

    def getVerticalScale(self, channel=1):
        return self.ask('CHAN%i:SCAL?' %(channel))

    def setVerticalPosition(self, channel=1, div=0):
        # vertical position of trace, divs
        self.write('CHAN%i:POS %.2f' %(channel, div))

    def getVerticalPosition(self, channel=1):
        return self.ask('CHAN%i:POS?' %(channel))

    def setVerticalOffset(self, channel=1, offset=0):
        # offset of trace, Volts
        self.write('CHAN%i:OFFS %.2f' %(channel, offset))

    def getVerticalOffset(self, channel=1):
        return self.ask('CHAN%i:OFFS?' %(channel))

    def checkClipping(self, index=1):
        # Checks if the index in question is clipping above the screen limit. Requires that the index passed is for the Vpp for a particular channel
        status = (self.getMeasurementResult(index) == '9.91E+37\n')
        return status

    def fixClipping(self, index=1, channel=1, scale=5e-3):
        # Channel to check and index that contains the Vpp for that particular channel
        self.setVerticalScale(channel=channel, div=scale)
        time.sleep(2)
        inscale = scale
        while self.checkClipping(index=index):
            inscale = self.getVerticalScale(channel=channel)
            inscale = 1.25*float(inscale[:-1])
            self.setVerticalScale(channel=channel, div=inscale)
            time.sleep(3)
            self.wait(10)
        return inscale

    def fixMathClipping(self, index=1, channel=1, scale=5e-3):
        # Channel to check and index that contains the Vpp for that particular channel
        self.setMathScale(index=channel, scale=scale)
        time.sleep(2)
        while self.checkClipping(index=index):
            inscale = self.getMathScale(index=channel)
            inscale = 1.25*float(inscale[:-1])
            self.setMathScale(index=channel, div=inscale)
            time.sleep(3)
            self.wait(10)
            
    ######################################################
    # ACQUISITION 
    ######################################################

    def setAcquisitionType(self, mode='REF'):
        # available modes: REFresh, AVERage, ENVelope
        self.write('ACQ:TYPE %s' %(mode))

    def getAcquisitionType(self):
        return self.ask('ACQ:TYPE?')

    def setAcquisitionAuto(self, mode='ON'):
        # available modes: ON, OFF
        self.write('ACQ:POIN:AUT %s' %(mode))

    def getAcquisitionAuto(self):
        return self.ask('ACQ:POIN:AUT?')

    def setAcquisitionPoints(self, points=100e3):
        # Set number of points in a waveform to record in a segment. Options are 5k samples to 80M samples.
        self.write('ACQ:POIN:VAL %.3f' %(points))

    def getAcquisitionPoints(self):
        return self.ask('ACQ:POIN:VAL?')

    def setAcquisitionMode(self, mode='AUT'):
        # Set how the record length is set: AUTomatic, DMEMory, MANual
        self.write('ACQ:MEM:MODE %s' %(mode))

    def getAcquisitionMode(self):
        return self.ask('ACQ:MEM:MODE?')

    ######################################################
    # TRIGGER 
    ######################################################

    def setTriggerMode(self, src='A', mode='NORM'):
        # available modes for A: AUTO, NORMal
        # available modes fo B : DELay, EVENts
        self.write('TRIG:%s:MODE %s' %(src, mode))
    
    def getTriggerMode(self, src='A'):
        return self.ask('TRIG:%s:MODE?' %(src))

    def setTriggerType(self, src='A', mode='EDGE'):
        # available modes A: EDGE, WID, TV, RUNT, LOG, BUS, RIS, LINE
        self.write('TRIG:%s:TYPE %s' %(src, mode))

    def getTriggerType(self, src='A'):
        return self.ask('TRIG:%s:TYPE?' %(src))

    def setTriggerSource(self, src='A', channel=1):
        self.write('TRIG:%s:SOUR CH%i' %(src, channel))

    def getTriggerSource(self, src='A'):
        return self.ask('TRIG:%s:SOUR?' %(src))

    def setTriggerEdgeCoupling(self, src='A', mode='DC'):
        # available modes: DC, LFR, AC
        self.write('TRIG:%s:EDGE:COUP %s' %(src, mode))

    def getTriggerEdgeCoupling(self, src='A'):
        return self.ask('TRIG:%s:EDGE:COUP?' %(src))

    def setTriggerEdgeSlope(self, src='A', mode='RISE'):
        # available modes: POSitive, NEGative, EITHer
        self.write('TRIG:%s:EDGE:SLOP %s' %(src, mode))

    def getTriggerEdgeSlope(self, src='A'):
        return self.ask('TRIG:%s:EDGE:SLOP?' %(src))

    def setTriggerEdgeLevel(self, channel=2, level=0):
        # trigger edge level, Volts, only available for trigger A
        self.write('TRIG:A:LEV%i:VAL %.2f' %(channel, level))

    def getTriggerEdgeLevel(self, channel=2):
        return self.ask('TRIG:A:LEV%i:VAL?' %(channel))

    def setTriggerAutoLevel(self):
        self.write('TRIG:A:FIND')

    ######################################################
    # TRIGGER B 
    ######################################################

    def setTriggerBDelayTime(self, time=100e-9):
        self.write('TRIG:B:DEL %.2e' %(time))

    def getTriggerBDelayTime(self):
        return self.ask('TRIG:B:DEL?')

    ######################################################
    # CURVE DATA 
    ######################################################

    def setDataSource(self, channel=1):
        self.write('EXP:WAV:SOUR CH%i' %(channel))

    def getDataSource(self):
        return self.ask('EXP:WAV:SOUR?')

    def setDataDestination(self, dest='/USB_FRONT/WFM'):
        self.write('EXP:WAV:NAME %s' %(dest))

    def getDataDestination(self):
        return self.ask('EXP:WAV:NAME?')

    def saveWaveformData(self):
        self.write('EXP:WAV:SAVE')

    def setDataFormat(self, form='CSV', bitvalue=0):
        # Set format of saved data: ASCii, REAL, UINTeger, CSV
        # Set accuracy of data: 0, 8, 16, 32 (0 is default for CSV; instrument decides)
        self.write('FORM %s,%i' %(form, bitvalue))

    def getDataFormat(self):
        return self.ask('FORM?')
    
    def getWaveformSampleRate(self):
        return self.ask('ACQ:SRAT?')
    
    def getSampleRate(self):
        return self.ask('ACQ:POIN:ARAT?')

    ######################################################
    # MEASUREMENT 
    ######################################################

    def startAquisition(self):
        self.write('START')

    def stopAquisition(self):
        self.write('STOP')
        
    def setMeasurement(self, index=1, mode='FREQ'):
         # available modes: see documentation at https://www.rohde-schwarz.com/webhelp/RTM3000_HTML_UserManual_en/Content/abf34e6145b64437.htm
        # Sets the measurement tracked in location 'index'
        self.write('MEAS%i:MAIN %s' %(index, mode))

    def getMeasurement(self, index=1):
        return self.ask('MEAS%i:MAIN?' %(index))

    def toggleMeasurement(self, index=1, state='ON'):
        # Set measurement in index ON or OFF
        self.write('MEAS%i %s' %(index, state))
    
    def setMeasurementSource(self, index=1, channel=1):
        self.write('MEAS%i:SOUR CH%i' %(index, channel))

    def setArbitraryMeasurementSource(self, index, source):
        self.write(f'MEAS{index}:SOUR {source}')
        
    def getMeasurementSource(self, index=1):
        return self.ask('MEAS%i:SOUR?' %(index, src))

    def toggleMeasurementStats(self, state='OFF'):
        self.write('MEAS:STAT %s' %(state))

    def resetMeasurementStats(self, ch=1):
        # Resets measurement statistics 
        self.write('MEAS%i:STAT:RES' %(ch))

    def toggleAutoMeasureTScale(self, state='ON'):
        self.write('MEAS1:TIM:AUTO')

    def setMeasureTScale(self, dt=200e-6):
        # manually set time to wait before measurement is returned. Should be atleast 12*horizontal_scale + trigger_period in seconds
        self.write('MEAS1:TIM %.3f' %(dt))

    def getMeasurementResult(self, index=1):
        return self.ask('MEAS%i:RES?' %(index))
        
    def getMeasurementAvg(self, index=1):
        return self.ask('MEAS%i:RES:AVG?' %(index))

    def getMeasurementStd(self, index=1):
        return self.ask('MEAS%i:RES:STDD?' %(index))

    ######################################################
    # AQUISITION
    ######################################################

    def setAquisitionType(self, aq='AVER'):
        self.write(f'ACQ:TYPE {aq}')

    def getAquisitionType(self):
        return self.ask('ACQ:TYPE?')

    def setAverageCount(self, val=1000):
        self.write(f'ACQ:AVER:COUN {val}')

    def getAverageCount(self):
        return self.ask('ACQ:AVER:CURR?')

    def setSampleMode(self, mode='SAMP'):
        self.write(f'CHAN:TYPE {mode}')

    def getSampleMode(self):
        return self.ask('CHAN:TYPE?')

    def setSampleState(self, state='OFF'):
        self.write(f'CHAN:ARIT {state}')
    
    ######################################################
    # CHANNEL SETUP
    ######################################################

    def toggleChannel(self, channel=1, status='ON'):
        self.write('CHAN%i:STAT %s' %(channel,status))

    def statusChannel(self, channel=1):
        return self.ask('CHAN%i:STAT?' %(channel))

    def setChanCoupling(self, channel=1, coup='DCLimit'):
        self.write('CHAN%i:COUP %s' %(channel,coup))

    def getChanCoupling(self, channel=1):
        self.ask('CHAN%i:COUP?' %(channel))
    
    ######################################################
    # WAVEFORM GENERATOR
    ######################################################

    def setWaveFunction(self, fun='SIN'):
        # Sets the generated waveform to DC, SINusoid, SQUare, PULSe, TRIangle, RAMP, SINC, ARBitrary, EXPonential
        self.write('WGEN:FUNC %s' %(fun))

    def getWaveFunction(self):
        return self.ask('WGEN:FUNC?')

    def setWaveVoltage(self, amp=2.5e-1):
        # Sets the waveform generated amplitude
        self.wavevolt = amp
        self.write('WGEN:VOLT %.2e' %(amp))

    def getWaveVoltage(self):
        return self.ask('WGEN:VOLT?')

    def setWaveVoltOffset(self, offset=0):
        # Sets the waveform offset
        self.write('WGEN:VOLT:OFFS %.2e' %(offset))

    def getWaveVoltOffset(self):
        return self.ask('WGEN:VOLT:OFFS?')

    def setWaveVoltFrequency(self, freq=1e4):
        # Sets the waveform frequency in Hz
        self.write('WGEN:FREQ %.2e' %(freq))

    def getWaveVoltFrequency(self):
        return self.ask('WGEN:FREQ?')

    def setWaveNoise(self, noise=0e0):
        # Sets noise in waveform in absolute volts
        self.write('WGEN:NOIS:ABS %.2e' %(noise))

    def getWaveNoise(self):
        return self.ask('WGEN:NOIS:ABS?')

    def toggleWaveform(self, status='OFF'):
        self.write('WGEN:OUTP %s' %(status))

    def getWaveformStatus(self):
        return self.ask('WGEN:OUTP?')

    def toggleWaveformBurst(self, status='ON'):
        self.write('WGEN:BURS %s' %(status))

    def getWaveformBurst(self):
        return self.ask('WGEN:BURS?')

    def setWaveformBurstCount(self, cycles=10):
        self.write('WGEN:BURS:NCYC %i' %(cycles))

    def getWaveformBurstCount(self):
        return self.ask('WGEN:BURS:NCYC?')

    def setWaveformBurstIdle(self, time=1e-1):
        self.write('WGEN:BURS:ITIM %.2e' %(time))

    def getWaveformBurstIdle(self):
        return self.ask('WGEN:BURS:ITIM?')

    def setStartFreqSweep(self, freq=10e3):
        self.write(f'WGEN:SWE:FST {freq}')

    def setEndFreqSweep(self, freq=10e4):
        self.write(f'WGEN:SWE:FEND {freq}')

    def setSweepTime(self, time=1):
        self.write(f'WGEN:SWE:TIME {time}')

    def setSweepType(self, style='LIN'):
        self.write(f'WGEN:SWE:TYPE {style}')

    def toggleSweep(self, toggle='OFF'):
        self.write(f'WGEN:SWE:ENAB {toggle}')
        
    def getWaveInfo(self):
        wavefunc = self.getWaveFunction()
        wavevolt = self.getWaveVoltage()
        wavefreq = self.getWaveVoltFrequency()
        return wavefunc, wavevolt, wavefreq
    
    def setWaveInfo(self, fun='SIN', amp=2.5e-1, offset=0, freq=1e4):
        self.setWaveFunction(fun)
        self.setWaveVoltage(amp)
        self.setWaveVoltOffset(offset)
        self.setWaveVoltFrequency(freq)

    ######################################################
    # FFT 
    ######################################################
        
    def enableSpec(self):
        self.write('SPEC:STAT ON')

    def disableSpec(self):
        self.write('SPEC:STAT OFF')

    def setSpecChan(self, channel=1):
        self.write(f'SPEC:SOUR CH{channel}')

    def setSpecWindowType(self, win='HANN'):
        self.write(f'SPEC:FREQ:WIND:TYPE {win}')

    def setSpecScaling(self, scale='DBM'):
        self.write(f'SPEC:FREQ:MAGN:SCAL {scale}')

    def setSpecFreqCenter(self, center=25e3):
        self.write(f'SPEC:FREQ:CENT {int(center)}')

    def setSpecFreqSpan(self, span=50e3):
        self.write(f'SPEC:FREQ:SPAN {int(span)}')

    def setSpecFreqStart(self, start=1e3):
        self.write(f'SPEC:FREQ:STAR {int(start)}')

#    def setSpecFreqStop(self, stop=50e3):
#        self.

    def getSpecWavData(self):
        self.ask('SPEC:WAV:SPEC:DATA?')

    ######################################################
    # MATHEMATICS
    ######################################################

    def setMathScale(self, index=1, scale=1):
        self.write(f'CALC:MATH{index}:SCAL {scale}')

    def getMathScale(self, index=1):
        return self.ask(f'CALC:MATH{index}:SCAL?')

    def subtractChannels(self, channel_one=1, channel_two=2, waveform=1):
        self.write(f'CALC:MATH{waveform}:EXPR:DEF "SUB(CH{channel_one},CH{channel_two}) in V"')

    def addChannels(self, channel_one=1, channel_two=2, waveform=1):
        self.write(f'CALC:MATH{waveform}:EXPR:DEF "ADD(CH{channel_one},CH{channel_two}) in V"')

    def filterLP(self, waveform=1, ref='M1', freq=10e4):
        self.write(f'CALC:MATH{waveform}:EXPR:DEF "LP({ref},{freq})"')

    def enableMath(self, index=1):
        self.write(f'CALC:MATH{index}:STAT ON')
        
    ######################################################
    # CUSTOM SEQUENCES 
    ######################################################

    def simpleSetup(self, burst=True, trig=0):
        channels = [1,2]
        for channel in channels:
            self.toggleChannel(channel=channel, status='ON')
            if channel == 2:
                self.setChanCoupling(channel=channel,coup='DCLimit')
            else:
                self.setChanCoupling(channel=channel,coup='ACLimit')
        self.simpleEdgeTrigger(level=trig)
        self.simpleWaveform(cycles=100, burst=burst)
        self.SimpleSetupStatus = True
        
    def simpleEdgeTrigger(self, channel=2, level=0, coupl='DC', mode='RISE'):
        self.setTriggerMode('A', 'NORM')
        self.setTriggerType('A', 'EDGE')
        self.setTriggerSource('A', channel)
        self.setTriggerEdgeCoupling('A', coupl)
        self.setTriggerEdgeSlope('A', mode)
        self.setTriggerEdgeLevel(channel, level)

    def simpleWaveform(self, fun='SIN', amp=1, offset=0, freq=1e4, delay=1e-1, cycles=5, burst=True, channel=2):
        self.toggleWaveform('ON')
        self.setWaveInfo(fun, amp, offset, freq)
        self.setVerticalScale(channel=channel, div=amp/3)
        self.setVerticalOffset(channel=channel, offset=0)
        if burst:
            self.toggleWaveformBurst('ON')
            self.setWaveformBurstCount(cycles)
            self.setWaveformBurstIdle(delay)


    def setSimpleMeasurements(self):
        self.setMeasurementSource(index=1, channel=1)
        self.setMeasurementSource(index=2, channel=1)
        self.setMeasurementSource(index=3, channel=1)
        self.setMeasurementSource(index=4, channel=1)
        self.setMeasurementSource(index=5, channel=2)
        self.setMeasurementSource(index=6, channel=2)
        self.setMeasurementSource(index=7, channel=2)
        self.setMeasurementSource(index=8, channel=2)
        self.setMeasurement(index=1, mode='PEAK')
        self.setMeasurement(index=5, mode='PEAK')        
        self.setMeasurement(index=2, mode='FREQ')
        self.setMeasurement(index=6, mode='FREQ')        
        self.setMeasurement(index=3, mode='MEAN')
        self.setMeasurement(index=7, mode='MEAN')
        self.SimpleMeasurementStatus = True

    def getSimpleMeasurements(self):
        peak1 = self.getMeasurementResult(index=1)
        peak2 = self.getMeasurementResult(index=5)
        freq1 = self.getMeasurementResult(index=2)
        freq2 = self.getMeasurementResult(index=6)
        mean1 = self.getMeasurementResult(index=3)
        mean2 = self.getMeasurementResult(index=7)
        return [peak1[:-1], peak2[:-1], freq1[:-1], freq2[:-1], mean1[:-1], mean2[:-1]]

    def getMeasurements(self, measures=8):
        data = []
        for i in range(measures):
            data.append(self.getMeasurementResult(index=i+1)[:-1])
        return data

    def getSimpleMean(self):
        peak1 = self.getMeasurementAvg(index=1)
        peak2 = self.getMeasurementAvg(index=5)
        freq1 = self.getMeasurementAvg(index=2)
        freq2 = self.getMeasurementAvg(index=6)
        mean1 = self.getMeasurementAvg(index=3)
        mean2 = self.getMeasurementAvg(index=7)
        return [peak1[:-1], peak2[:-1], freq1[:-1], freq2[:-1], mean1[:-1], mean2[:-1]]

    def getSimpleSTD(self):
        peak1 = self.getMeasurementStd(index=1)
        peak2 = self.getMeasurementStd(index=5)
        freq1 = self.getMeasurementStd(index=2)
        freq2 = self.getMeasurementStd(index=6)
        mean1 = self.getMeasurementStd(index=3)
        mean2 = self.getMeasurementStd(index=7)
        return [peak1[:-1], peak2[:-1], freq1[:-1], freq2[:-1], mean1[:-1], mean2[:-1]]

    
    def setSimpleScale(self):
        if not self.SimpleSetupStatus:
            print("simpleSetup is not set and simple scaling can't be done.")
            return 0
        amp_scale = self.wavevolt#*0.25
        self.fixClipping(index=5, channel=2, scale=amp_scale/2)#5e-1) # scale should be set as 1 for every 4 V on the generated wavevoltage
        self.fixClipping(index=1, channel=1, scale=0.01)#*1e-2)
        return 1

    def fullResetStats(self):
        for i in range(8):
            ch = i + 1
            self.resetMeasurementStats(ch=ch)

#    def getSimpleWaveform(self):
        
        
