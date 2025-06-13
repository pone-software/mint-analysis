import time
import serial

class K6482(serial.Serial):
    '''This class creates an instance for the Keithley 6485 Picoammeter 
    as a USB instrument.
    '''
    def __init__(self, device='PICOAMP', baudrate=57600, timeout=2):
        super().__init__('/dev/%s' %(device), baudrate=baudrate, timeout=timeout)
        print('Initializing Keithley 6482 picoammeter...')
        self.start()
        print('... done.\n')

    def writeCmd(self, cmd):
        self.write(str.encode('%s\r' %(cmd)))
        time.sleep(0.5)
    
    def writeList(self, aList):
        for i in aList:
            self.write(str.encode('%s\r' %(i)))
            time.sleep(0.5)

    def identify(self):
        self.writeCmd('*IDN?\r')
        line = self.readline()
        return line.decode()
    
    def reset(self):
        self.writeCmd('*RST')

    def setRemote(self):
        self.writeCmd('SYST:REM')

    def setAutoRange(self, ch=1, mode='ON'):
        self.writeCmd('SENS%i:CURR:RANG:AUTO %s' %(ch, mode))

    def setAutozero(self):
        self.writeCmd('SYST:AZERO:STAT ON')

    def start(self):
        self.reset()
        self.setRemote()
        self.setAutoRange(ch=1)
        self.setAutoRange(ch=2)
        self.setAutozero()

    def readCurrent(self, ch='CH1', n=1):
        reading = ['ARM:SOUR IMM', 'ARM:COUN 1', 'TRIG:SOUR IMM',
                   'TRIG:COUN %i' %(n), 'READ?']
        self.writeList(reading)
        line = self.readline()
        vals_clean = list(filter(None, line.decode('utf8').split(',')))
        vals = [float(i) for i in vals_clean]
        vals_ch1 = vals[0::2]
        vals_ch2 = vals[1::2]
        if ch == 'CH1':
            return vals_ch1
        if ch == 'CH2':
            return vals_ch2
        if ch == 'BOTH':
            return vals
