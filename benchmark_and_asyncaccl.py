import numpy as np
from memryx import Benchmark
from memryx import AsyncAccl

import time
dfp_filename = './dfp_models/yolov5n-SiLU-416.dfp'
resolution = 416
num_frames = 1000
count = 0

def data_source():
    for n in range(num_frames):
        img = np.zeros((resolution, resolution, 3), dtype=np.float32)
        yield img

def output_processor(*outputs):
    global count
    count += 1



def run_test(use_bm=False):

    global count
    count = 0
    if use_bm:
        with Benchmark(dfp=dfp_filename) as bm:
            _, latency_ms, _ = bm.run(threading=False)
        with Benchmark(dfp=dfp_filename) as bm:
            _, _, fps_bm = bm.run(frames=num_frames, threading=True)
    else:
        latency_ms = 0
        fps_bm = 0

    time.sleep(2)
    accl = AsyncAccl(dfp_filename, chip_gen=3.1)

    t0 = time.perf_counter()
    accl.connect_input(data_source)
    accl.connect_output(output_processor)
    accl.wait()
    t1 = time.perf_counter()

    accl.stop()
    # shutdown seems to be needed here when alternating between Benchmark and AsyncAccl
    accl.shutdown()
    time.sleep(2)

    dt = t1 - t0
    fps_aa = int(count / dt)

    print(f'FPS_bm: {int(fps_bm)}, FPS_aa: {fps_aa}, Latency: {latency_ms}')



if __name__ == '__main__':

    print('\nUsing Benchmark and AsyncAccl:')
    for n in range(8):
        run_test(use_bm=True)

    # print('\nUsing only AsyncAccl:')
    # for n in range(8):
    #     run_test(use_bm=False)


'''
1. Run both tests back to back without exiting python script:

Using Benchmark and AsyncAccl:
FPS_bm: 434, FPS_aa: 434, Latency: 10.252084732055664
FPS_bm: 434, FPS_aa: 434, Latency: 10.215494632720947
FPS_bm: 434, FPS_aa: 156, Latency: 10.00978946685791
FPS_bm: 434, FPS_aa: 434, Latency: 10.020782947540283
FPS_bm: 434, FPS_aa: 157, Latency: 10.379102230072021
FPS_bm: 434, FPS_aa: 434, Latency: 10.478935241699219
FPS_bm: 434, FPS_aa: 156, Latency: 10.50675630569458
FPS_bm: 434, FPS_aa: 434, Latency: 10.050656795501709

Using only AsyncAccl:
FPS_bm: 0, FPS_aa: 434, Latency: 0
FPS_bm: 0, FPS_aa: 156, Latency: 0
FPS_bm: 0, FPS_aa: 434, Latency: 0
FPS_bm: 0, FPS_aa: 434, Latency: 0
FPS_bm: 0, FPS_aa: 435, Latency: 0
FPS_bm: 0, FPS_aa: 156, Latency: 0
FPS_bm: 0, FPS_aa: 434, Latency: 0
FPS_bm: 0, FPS_aa: 157, Latency: 0


2. Run only AsyncAccl test:

Using only AsyncAccl:
FPS_bm: 0, FPS_aa: 434, Latency: 0
FPS_bm: 0, FPS_aa: 434, Latency: 0
FPS_bm: 0, FPS_aa: 434, Latency: 0
FPS_bm: 0, FPS_aa: 434, Latency: 0
FPS_bm: 0, FPS_aa: 433, Latency: 0
FPS_bm: 0, FPS_aa: 434, Latency: 0
FPS_bm: 0, FPS_aa: 434, Latency: 0
FPS_bm: 0, FPS_aa: 433, Latency: 0

'''