import serial
import time
import queue
import threading
import numpy as np

'''
Get the CH341 driver from here: https://github.com/WCHSoftGroup/ch341ser_linux
and follow instructions to build and install.

If device is not showing up check dmesg. I had a conflict with brltty which was being loaded
because it was listed in a udev rules with the same vendor and product ID. I removed it.

Documentation here:
https://elmorlabs.com/product/elmorlabs-pmd-usb-power-measurement-device-with-usb/

'''

class PMDReader():
    pmd_settings = {
        'port':'/dev/ttyCH341USB0',
        'baudrate':115200,
        'bytesize':8,
        'stopbits':1,
        'timeout':1
        }
    
    def __init__(self, port, sensor_name):
        try:
            self.pmd_settings['port'] = port
            self.ser = serial.Serial(**self.pmd_settings)
        except serial.SerialException:
            print(f"{self.pmd_settings['port']} not able to connect")
            self.ser = None
        self.running = False
        self.sensors = ['PCIE1', 'PCIE2', 'EPS1', 'EPS2']
        assert sensor_name in self.sensors
        self.selected_sensor = sensor_name
        self.dt_s = 0.5  # time in seconds to pause in between readings


    def start_reading(self):
        self.running = True
        self.power_q = queue.Queue()
        self.worker_thread = threading.Thread(target=self.sensor_reader)
        self.worker_thread.start()
        print('Power reading started')

    def stop_reading(self):
        self.running = False
        self.worker_thread.join()
        print('Power reading stopped')

    def avg_recent_readings(self):
        '''
        return the average of recent readings
        '''
        vs = [self.power_q.get()]
        while not self.power_q.empty():
            vs.append(self.power_q.get())
        vs = np.array(vs)
        print(vs)
        print(vs.min())
        print(vs.max())
        print(vs.mean())
        return vs.mean()


    def sensor_reader(self):
        '''
        read the sensor at interval dt_s and put readings in the queue
        '''
        while self.running:
            power = self.get_new_sensor_values()[self.selected_sensor]
            self.power_q.put(power)
            time.sleep(self.dt_s)

    def check_device(self):
        # b'\x00'   welcome message
        # b'\x01'   ID
        # b'\x02'   read sensors
        # b'\x03'   read values
        # b'\x04'   read config
        # b'\x06'   read ADC buffer

        # check welcome message
        self.ser.write(b'\x00')
        self.ser.flush()
        read_bytes = self.ser.read(18)
        device_detected = (read_bytes == b'ElmorLabs PMD-USB')
        return device_detected

    def get_new_sensor_values(self):
        command = b'\x03'
        self.ser.write(command)
        self.ser.flush()
        read_bytes = self.ser.read(16)

        power_vals = {}

        for i, name in enumerate(self.sensors):
            # convert bytes to float values
            voltage_value = int.from_bytes(read_bytes[i*4:i*4+2], byteorder='little')*0.01
            current_value = int.from_bytes(read_bytes[i*4+2:i*4+4], byteorder='little')*0.1
            power_value = voltage_value * current_value
            power_vals[self.sensors[i]] = power_value

        return power_vals
    

def main():
    pmd = PMDReader()
    pmd.check_device()

    for i in range(10):
        pmd.get_new_sensor_values()
        print()
        time.sleep(1)
    
if __name__ == '__main__':
    main()
