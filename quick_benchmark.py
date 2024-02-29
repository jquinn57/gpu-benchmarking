from memryx import Benchmark
import numpy as np

dfp_filename= './dfp_models/yolov5s-LeakyReLU-416.dfp'


for n in range(10):
    with Benchmark(dfp=dfp_filename, verbose=2, chip_gen=3.1) as accl:
        # 1000 frames, get FPS
        outputs,_,fps = accl.run(frames=1000)

        # single frame, get latency
        outputs,latency,_ = accl.run(threading=False)
        print(f'FPS: {fps}, latency: {latency}')


