import time
import serial
class Pulser(serial.Serial):
    '''This class creates an instance for the Water Monitor Board by N. Braahms 
    as a USB instrument.
    '''
    def __init__(self, device='PULSER', baudrate=115200, timeout=2):
        super().__init__('/dev/serial/by-id/usb-FTDI_FT230X_Basic_UART_DP074IL4-if00-port0',
                         baudrate=baudrate,
                         bytesize=serial.EIGHTBITS,
                         parity=serial.PARITY_NONE,
                         stopbits=serial.STOPBITS_ONE,
                         inter_byte_timeout=0.1,
                         timeout=timeout)
        print('Initializing Pulser Board...')
        self.start()
        print('... done.\n')

    def writeCmd(self, cmd):
        #print('\t\t-- sending: ', cmd)
        cmd_enc = str.encode('%s' %(cmd))
        self.write(cmd_enc)
        time.sleep(0.1)
    
    def writeList(self, aList):
        for i in aList:
            self.writeCmd(i)
            time.sleep(0.1)

    def readAll(self):
        lines = ''
        while True:
            line = self.readline().decode()
            if line != '':
                lines += line
            else:
                break
        if lines != '': 
            lines = lines.replace('\n\r', '\r\n')
            ret = lines.lstrip().split('\r\n')
        else:
            ret = ['']
        return [i for i in ret if i != '']

    def printRead(self):
        ret = self.readAll()
        for i in ret:
            print('\t\t-- %s' %(i))
        
    def start(self):
        self.enable()
        self.selectLED(0)

    def enable(self):
        self.writeCmd('E')
        self.printRead()

    def disable(self):
        self.writeCmd('D')
        self.printRead()

    def selectTrigger(self, mode='internal'):
        # internal or external trigger (3.3V/50Ohms)
        x = 'E' if mode == 'external' else 'I' if mode == 'internal' else ''
        self.writeCmd('T%s' %(x))
        self.printRead()

    def selectTriggerRateInternal(self, rate='1'):
        # slow (1.5kHz) or fast (8MHz) internal trigger rate
        x = 'S' if rate == 'slow' else 'F'
        self.writeCmd('R%s' %(rate))
        self.printRead()

    def selectLED(self, led=0):
        # select LEDs 1 - 7; 0 switches all LEDs off
        self.writeCmd('L%i' %(led))
        self.printRead()

    def setBiasVoltage(self, vbias=5.0):
        # set bias voltage dac in volts
        dac = int((vbias-14.7) / (-12.1/1023))
        send = 'S%s' %(str(dac).zfill(4))
        self.writeCmd(send)
        time.sleep(5)
        self.printRead()

    def readBiasAdc(self):
        self.writeCmd('Q')
        ret = self.readAll()
        print(ret)
        adc = int(ret[0].split('=')[1].lstrip()) 
        v   = (16.5/4095)*adc
        return v

    def flash(self, led=0, vbias=5.0, trigger='internal',
              rate='fast'):
        self.enable()
        self.selectTrigger(mode=trigger)
        if trigger == 'internal':
            self.selectTriggerRateInternal(rate=rate)
        self.setBiasVoltage(vbias=vbias)
        print('\t\t-- Bias ADC: ', self.readBiasAdc())
        self.selectLED(led)
