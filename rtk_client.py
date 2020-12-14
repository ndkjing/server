from serialData.serial_data import SerialData
from tcpServer.tcp_client import TcpClient
import threading

class RtkClient:
    def __init__(self):
        self.serial_data_obj = SerialData()
        self.tcp_client_obj = TcpClient()

    def run(self):
        serial_thread = threading.Thread(target=self.serial_data_obj.read)
        # self.serial_data_obj.read()
        serial_thread.start()

        while True:
            print('self.serial_data_obj.data:',self.serial_data_obj.data)
            self.tcp_client_obj.send_data(self.serial_data_obj.data)
        serial_thread.join()

if __name__ == '__main__':
    obj = RtkClient()
    obj.run()

















