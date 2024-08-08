
import asyncio
import threading
import time
from queue import Queue
from kasa import SmartStrip, SmartPlug
import numpy as np
import json
# pip install python-kasa
# run on command line to discover devices on local network: kasa discover



class KasaReader:
    def __init__(self, ip, plug_name):
        self.ip = ip
        self.plug_name = plug_name
        self.power_q = Queue()
        self.power_q_hist = Queue()
        self.event_q = Queue()

        self.dt_s = 0.5   # time interval in seconds between readings
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, args=(self.loop,))
        self.running = False

    def _run_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    def start_reading(self):
        # throw away old readings in case there was a delay after previous test
        while not self.power_q.empty():
            self.power_q.get()
        if not self.running:
            print('Kasa thread starting')
            self.running = True
            self.thread.start()
            asyncio.run_coroutine_threadsafe(self.read_power(), self.loop)
            time.sleep(4)
        self.mark_event()


    def mark_event(self):
        print('Mark event')
        self.event_q.put(time.perf_counter())


    def stop_reading(self):
        if self.running:
            print('Kasa thread stoping')
            time.sleep(4)
            self.running = False
            self.loop.call_later(2 * self.dt_s, self.loop.stop)
            self.thread.join()
        
        power_ts = []
        while not self.power_q_hist.empty():
            power_ts.append(self.power_q_hist.get())
        events = []
        while not self.event_q.empty():
            events.append(self.event_q.get())
        output = {'events': events, 'power_ts': power_ts}
        with open('power_hist.json', 'wt') as fp:
            json.dump(output, fp)

    
    def avg_recent_readings(self):
        '''
        return the average of recent readings
        '''
        vs = [self.power_q.get()]
        while not self.power_q.empty():
            vs.append(self.power_q.get())
        vs = np.array(vs)
        if len(vs) > 2:
            vs = vs[1:-1]
        print(vs)
        return vs.mean()


    async def read_power(self):
        ss = SmartStrip(self.ip)
        await ss.update()
        sp = ss.get_plug_by_name(self.plug_name)
        await asyncio.sleep(self.dt_s)
        while self.running:
            power = await sp.get_emeter_realtime()
            self.power_q.put(power.power)
            ts = time.perf_counter()
            self.power_q_hist.put((ts, power.power))
            # print(power.power)
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

