import yaml
import argparse
import pprint
import onnxruntime
import time
import numpy as np
import cv2
import glob
import os
import math
import threading
import queue
from pmd_reader import PMDReader
from kasa_reader import KasaReader

class GPUBenchmark():
    def __init__(self, settings):
        self.settings = settings
        # TODO: add option to read power via nvidia-smi
        self.pmd_reader = None
        if 'pmd' in settings:
            self.pmd_reader = PMDReader(*settings['pmd'])
            if not self.pmd_reader.check_device():
                print('PMD not found')
                self.pmd_reader = None
        self.kasa_reader = None
        if 'kasa' in settings:
            self.kasa_reader = KasaReader(*settings['kasa'])

            
    def image_worker(self, res):
        n = 0
        while self.running and n < self.num_images:
            img_filename = self.image_list[n]
            img = cv2.imread(img_filename)
            img = cv2.resize(img, (res, res)).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            self.image_q.put(img)
            n += 1

    def run_test(self, model_config, batch_size, do_inference=True):
        '''
        Run a test with one condition
        '''

        if 'image_path' in self.settings and self.settings['image_path']:
            self.load_images = True
            self.image_list = glob.glob(self.settings['image_path'] + '/*.jpg')
            self.num_images = min(len(self.image_list), self.settings['num_images'])
            self.image_q = queue.Queue()
            self.running = False
        else: # use random data instead of actual images
            self.load_images = False
            self.image_list = []
            self.num_images = self.settings['num_images']
        
        if do_inference:
            use_cuda = self.settings['onnx_ep'] == 'cuda'
            providers = ['CUDAExecutionProvider'] if use_cuda else  ['TensorrtExecutionProvider']
            session = onnxruntime.InferenceSession(model_config['filename'], providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            print(output_names)
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            print(input_name)
            print(input_shape)

        res = model_config['resolution']

        is_warm = False
        nbatches = int(math.floor(self.num_images / batch_size))
        count = 0

        if self.load_images:
            self.thread = threading.Thread(target=self.image_worker, args=(model_config['resolution'], ))
            self.running = True
            self.thread.start()

        if self.pmd_reader:
            self.pmd_reader.start_reading()
        if self.kasa_reader:
            self.kasa_reader.start_reading()

        t0 = time.perf_counter()
        for n in range(nbatches):

            if self.load_images:
                img_batch = np.zeros((batch_size, 3, res, res), dtype=np.float32)
                for i in range(batch_size):
                    img_batch[i, :, :, :] = self.image_q.get()
            else:
                img_batch = np.random.random((batch_size, 3, res, res)).astype(np.float32)
            
            if do_inference:
                y = session.run(output_names, {input_name: img_batch})[0]
            else:
                time.sleep(0.01)
            
            count += batch_size
            # restart the counter and timer after warmup batch
            if not is_warm:
                count = 0
                is_warm = True
                t0 = time.perf_counter()

        dt = time.perf_counter() - t0
        time_per_image = 1000 * dt / count
        fps = int(round(count / dt))

        if self.pmd_reader:
            self.pmd_reader.stop_reading()
            power_avg_pmd = int(round(self.pmd_reader.avg_recent_readings()))
        else:
            power_avg_pmd = 0
        
        if self.kasa_reader:
            self.kasa_reader.stop_reading()
            power_avg_kasa = int(round(self.kasa_reader.avg_recent_readings()))
        else:
            power_avg_kasa = 0
        
        print(f'Batch size: {batch_size}, Time per image: {dt} / {count} = {time_per_image} ms, FPS: {fps}, Power (PMD): {power_avg_pmd} W, Power (Kasa): {power_avg_kasa} W')
        return fps, power_avg_pmd, power_avg_kasa



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config yaml', default='config_gpu.yaml')
    parser.add_argument('--output_csv', help='Path to output csv', default='output_gpu.csv')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)
    gpu_benchmark = GPUBenchmark(config['settings'])

    output = ['Model, Resolution, Batch Size, FPS, Power(PMD), Power(Kasa)']
    for model in config['models']:
        print()
        print(model)
        model_config = config['models'][model]
        res = model_config['resolution']
        for batch_size in model_config['batch_sizes']:
            fps, power_pmd, power_kasa = gpu_benchmark.run_test(model_config, batch_size, do_inference=True)
            output.append(f'{model}, {res}, {batch_size}, {fps}, {power_pmd}, {power_kasa}')
    
    with open(args.output_csv, 'wt') as fp:
        for line in output:
            fp.write(line + '\n')


if __name__ == '__main__':
    main()