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
from memryx import AsyncAccl

class MX3Benchmark():
    def __init__(self, settings):
        self.settings = settings
        self.power_reader = None
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
        for img_filename in self.image_list[:self.num_images]:
            img = cv2.imread(img_filename)
            img = cv2.resize(img, (self.resolution, self.resolution)).astype(np.float32)
            yield img
    
    def output_processor(self, *outputs):
        self.count += 1


    def run_test(self, model_config):
        # Accelerate using the MemryX hardware
        dfp_filename = model_config['filename']
        self.resolution = model_config['resolution']
        accl = AsyncAccl(dfp_filename)
        #model_dir = os.path.dirname(dfp_filename)
        #dfp_basename = os.path.basename(dfp_filename)
        #onnx_filename = os.path.join(model_dir, 'model_0_' + dfp_basename.replace('.dfp', '_post.onnx'))
        if 'post_processing' in model_config:
            onnx_filename = model_config['post_processing']
            if os.path.exists(onnx_filename):
                print(f'Including post processing {onnx_filename}')
                accl.set_postprocessing_model(onnx_filename)

        accl.connect_input(self.data_source) # starts asynchronous execution of input generating callback
        accl.connect_output(self.output_processor) # starts asynchronous execution of output processing callback

        self.count = 0
        t0 = time.perf_counter()
        accl.wait() # wait for the accelerator to finish execution
        t1 = time.perf_counter()

        fps = self.count / (t1 - t0)
        inference_time_ms = 1000 * (t1 - t0) / self.count
        print(f'frame count: {self.count}')
        print(f'inference time ms: {inference_time_ms}')
        print(f'FPS: {fps}')
        return fps, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config yaml', default='config_mx3.yaml')
    parser.add_argument('--output_csv', help='Path to output csv', default='output_mx3.csv')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)
    mx3_benchmark = MX3Benchmark(config['settings'])

    output = ['Model, Resolution, Batch Size, FPS, Power']
    for model in config['models']:
        print()
        print(model)
        model_config = config['models'][model]
        res = model_config['resolution']
        batch_size = 1
        fps, power = mx3_benchmark.run_test(model_config)
        output.append(f'{model}, {res}, {batch_size}, {fps}, {power}')
    
    with open(args.output_csv, 'wt') as fp:
        for line in output:
            fp.write(line + '\n')


if __name__ == '__main__':
    main()