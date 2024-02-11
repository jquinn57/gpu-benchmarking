
import asyncio
import threading
import time
from queue import Queue
from kasa import SmartStrip, SmartPlug
import numpy as np
# pip install python-kasa
# run on command line to discover devices on local network: kasa discover



class KasaReader:
    def __init__(self, ip, plug_name):
        self.ip = ip
        self.plug_name = plug_name
        self.power_q = Queue()
        self.dt_s = 0.5   # time interval in seconds between readings
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, args=(self.loop,))
        self.running = False

    def _run_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    def start_reading(self):
        print('Kasa thread starting')
        if not self.running:
            self.running = True
            self.thread.start()
            asyncio.run_coroutine_threadsafe(self.read_power(), self.loop)

    
    def stop_reading(self):
        print('Kasa thread stoping')
        if self.running:
            self.running = False
            self.loop.call_later(2 * self.dt_s, self.loop.stop)
            self.thread.join()
    
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


    async def read_power(self):
        ss = SmartStrip(self.ip)
        await ss.update()
        sp = ss.get_plug_by_name(self.plug_name)
        await asyncio.sleep(self.dt_s)
        while self.running:
            power = await sp.get_emeter_realtime()
            self.power_q.put(power.power)
            await asyncio.sleep(self.dt_s)


def main():
    kasa_ip = '192.168.1.87'
    plug_name = 'workstation'
    kasa_reader = KasaReader(kasa_ip, plug_name)
    kasa_reader.start_reading()
    time.sleep(10)
    kasa_reader.stop_reading()
    print('done')
    avg = kasa_reader.avg_recent_readings()


if __name__ == '__main__':
    main()

