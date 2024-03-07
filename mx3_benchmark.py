# source ~/mx3-env/bin/activate
import yaml
import argparse
import pprint
import time
import numpy as np
import cv2
import glob
import os
import math
import threading
import queue
from kasa_reader import KasaReader
from memryx import AsyncAccl
from memryx import Benchmark

class MX3Benchmark():
    def __init__(self, settings):
        self.settings = settings
        self.do_post_processing = settings.get('post_processing', False)

        self.kasa_reader = None
        if 'kasa' in settings:
            self.kasa_reader = KasaReader(*settings['kasa'])

        self.count = 0
        if 'image_path' in self.settings and self.settings['image_path']:
            self.load_images = True
            self.image_list = glob.glob(self.settings['image_path'] + '/*.jpg')
            self.num_images = min(len(self.image_list), self.settings['num_images'])
        else: # use random data instead of actual images
             self.load_images = False
             self.image_list = []
             self.num_images = self.settings['num_images']


    def data_source(self):
        if self.load_images:
            for img_filename in self.image_list[:self.num_images]:
                img = cv2.imread(img_filename)
                img = cv2.resize(img, (self.resolution, self.resolution)).astype(np.float32)
                yield img
        else:
            for n in range(self.num_images):
                # img = np.random.random((self.resolution, self.resolution, 3)).astype(np.float32)
                # On Yolov5s-leakyrelu-640x640 using zeros: FPS = 100 
                # using random data FPS=80 -- surpisingly makes a big difference even though this runs in a seperate thread
                img = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)
                yield img
        
    def output_processor(self, *outputs):
        self.count += 1


    def baseline_power_test(self, model_config):
        dfp_filename = model_config['filename']
        self.resolution = model_config['resolution']
        if self.kasa_reader:
            self.kasa_reader.start_reading()
        else:
            return 0
        cnt = 0
        for img in self.data_source():
            cnt += 1

        power_avg_kasa = int(round(self.kasa_reader.avg_recent_readings()))
        return power_avg_kasa

    def run_test(self, model_config):

        if self.kasa_reader:
            self.kasa_reader.start_reading()
        dfp_filename = model_config['filename']
        self.resolution = model_config['resolution']
        accl = AsyncAccl(dfp_filename, chip_gen=3.1)

        if self.do_post_processing:
            model_dir = os.path.dirname(dfp_filename)
            dfp_basename = os.path.basename(dfp_filename)
            onnx_filename = os.path.join(model_dir, 'model_0_' + dfp_basename.replace('.dfp', '_post.onnx'))
            if os.path.exists(onnx_filename):
                print(f'Including post processing {onnx_filename}')
                accl.set_postprocessing_model(onnx_filename)

        accl.connect_input(self.data_source) # starts asynchronous execution of input generating callback
        accl.connect_output(self.output_processor) # starts asynchronous execution of output processing callback

        self.count = 0
        print(f'Start test of {dfp_filename}')
        t0 = time.perf_counter()
        accl.wait() # wait for the accelerator to finish execution
        t1 = time.perf_counter()
        # undocumented method?
        accl.shutdown()
        print('Done')

        if self.kasa_reader:
            power_avg_kasa = int(round(self.kasa_reader.avg_recent_readings()))
        else:
            power_avg_kasa = 0

        dt = t1 - t0
        fps = self.count / dt
        inference_time_ms = 1000 * dt  / self.count
        
        output = {}
        output['batch_size'] = 1
        output['inference_time_ms'] = inference_time_ms
        output['fps'] = fps
        output['system_power'] = power_avg_kasa
        output['pcie_power'] = 0
        output['count'] = self.count
        output['total_time_s'] = dt
        output['latency_ms'] = 0
        pprint.pprint(output)
        return output
    

    def shutdown(self):
        if self.kasa_reader:
            self.kasa_reader.stop_reading()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config yaml', default='config_mx3.yaml')
    parser.add_argument('--output_csv', help='Path to output csv', default='output_mx3.csv')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)
    mx3_benchmark = MX3Benchmark(config['settings'])

    outputs = ['Model, Resolution, Batch Size, FPS, Latency(ms), FPS (bm), System Power']

    for model in config['models']:
        print()
        print(model)
        model_config = config['models'][model]
        res = model_config['resolution']
        batch_size = 1
        #baseline_power = mx3_benchmark.baseline_power_test(model_config)
        #outputs.append(f"{model}, {res}, {0}, {0}, {0}, {0}, {baseline_power}")
        #print(f'Baseline power: {baseline_power}')
        data = mx3_benchmark.run_test(model_config)
        
        # Do a seperate latency measurement
        with Benchmark(dfp=model_config['filename'], verbose=2, chip_gen=3.1) as bm:
          _, data['latency_ms'], _ = bm.run(threading=False)
          _, _, fps_bm = bm.run(frames=1000)

        outputs.append(f"{model}, {res}, {batch_size}, {data['fps']}, {data['latency_ms']}, {fps_bm}, {data['system_power']}")
    
    mx3_benchmark.shutdown()
    with open(args.output_csv, 'wt') as fp:
        for line in outputs:
            fp.write(line + '\n')


if __name__ == '__main__':
    main()