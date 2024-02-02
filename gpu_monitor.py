import subprocess
import time
import signal
import os
import threading
import queue
import numpy as np

class Monitor():
    def __init__(self):

        # monitor the GPU utilization stats
        # on some GPUs this method could also be used to monitor power usage, but unfortunatly not on the RTX 4060
        command_str = "nvidia-smi dmon -i 0  -s u -c -1"
        command = command_str.split()
        print(command)
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.read_output)
        self.running = True
        self.thread.start()

    def read_output(self):
        
        while self.running:
            output = self.process.stdout.readline()
            row = output.decode().split()
            if row[0] == '0':
                vals = list(map(int, row))
                self.queue.put(vals)

    def stop(self):
        self.running = False
        self.thread.join()
        # Send SIGINT signal to the process group to terminate it
        os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        self.process.wait()

    def get_vals(self):
        vals = []
        while not self.queue.empty():
            vals.append(self.queue.get())
        vals_np = np.array(vals)
        return vals_np


if __name__ == '__main__':
    mon = Monitor()
    
    for i in range(5):
        time.sleep(4)
        print(mon.get_vals())
    
    mon.stop()
    print('done')