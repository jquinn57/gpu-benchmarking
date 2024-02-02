import serial
import time

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
    
    def __init__(self):
        try:
            self.ser = serial.Serial(**self.pmd_settings)
        except serial.SerialException:
            print(f"{self.pmd_settings['port']} not able to connect")
            self.ser = None
        
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
        assert read_bytes == b'ElmorLabs PMD-USB'

        # check sensor struct
        self.ser.write(b'\x02')
        self.ser.flush()
        read_bytes = self.ser.read(100)
        print('Struct: ', read_bytes)

    def get_new_sensor_values(self):
        command = b'\x03'
        self.ser.write(command)
        self.ser.flush()
        read_bytes = self.ser.read(16)

        sensors = ['PCIE1', 'PCIE2', 'EPS1', 'EPS2']
        power_vals = []

        for i, name in enumerate(sensors):

            # convert bytes to float values
            voltage_value = int.from_bytes(read_bytes[i*4:i*4+2], byteorder='little')*0.01
            current_value = int.from_bytes(read_bytes[i*4+2:i*4+4], byteorder='little')*0.1
            power_value = voltage_value * current_value
            print(f'{sensors[i]}: {power_value} Watts')
            power_vals.append(power_value)

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
