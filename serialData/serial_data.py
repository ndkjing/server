import serial, time
import binascii
import socket
import time

class SerialData:
    def __init__(self, com='com10', bot=38400, timeout=0.5):
        self.com = com
        self.bot = bot
        self.serial_com = serial.Serial(self.com, self.bot, timeout=timeout)
        self.data=None

    def read(self):
        while True:
            row_data = str(binascii.b2a_hex(self.serial_com.read(100)))
            if len(row_data)>3:
                print('time',time.time())
                print('row_data', len(row_data),row_data)
                self.data=row_data

if __name__ == '__main__':
    obj = SerialData()
    obj.read()
