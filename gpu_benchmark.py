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

class GPUBenchmark():
    def __init__(self, settings):
        self.settings = settings
        self.image_q = queue.Queue()
        self.running = False



    def image_worker(self, res):
        n = 0
        while self.running and n < self.num_images:
            img = cv2.imread(img_filename)
            img = cv2.resize(img, (res, res)).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            self.image_q.put(img)
            n += 1

    def run_test(self, model_config, batch_size):
        '''
        Run a test with one condition
        '''
        if 'image_path' in self.settings and self.settings['image_path']:
            self.load_images = True
            self.image_list = glob.glob(self.settings['image_path'] + '/*.jpg')
            self.num_images = min(len(image_list), self.settings['num_images'])
        else: # use random data instead of actual images
            self.load_images = False
            self.image_list = []
            self.num_images = self.settings['num_images']
        
        self.thread = threading.Thread(target=self.image_worker, model_config['resolution'])
        self.running = True
        self.thread.start()



def run_test(settings, model_config):

    use_cuda = settings['onnx_ep'] == 'cuda'
    providers = ['CUDAExecutionProvider'] if use_cuda else  ['TensorrtExecutionProvider']
    session = onnxruntime.InferenceSession(model_config['filename'], providers=providers)
    output_names = [x.name for x in session.get_outputs()]
    print(output_names)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(input_name)
    print(input_shape)
    res = model_config['resolution']
    fps_batch = {}

    if 'image_path' in settings and settings['image_path']:
        load_images = True
        image_list = glob.glob(settings['image_path'] + '/*.jpg')
        num_images = min(len(image_list), settings['num_images'])
    else: # use random data instead of actual images
        load_images = False
        image_list = []
        num_images = settings['num_images']

    for batch_size in model_config['batch_sizes']:
        is_warm = False
        nbatches = int(math.floor(num_images / batch_size))
        count = 0

        t0 = time.perf_counter()
        for n in range(nbatches):

            if load_images:
                img_batch = load_image_batch(image_list[count:count + batch_size], res)
            else:
                img_batch = np.random.random((batch_size, 3, res, res)).astype(np.float32)
            y = session.run(output_names, {input_name: img_batch})[0]
            count += batch_size
            # restart the counter and timer after warmup batch
            if not is_warm:
                count = 0
                is_warm = True
                t0 = time.perf_counter()



        dt = time.perf_counter() - t0

        time_per_image = 1000 * dt / count
        fps = count / dt
        fps_batch[batch_size] = fps
        print(f'Batch size: {batch_size}, Time per image: {time_per_image} ms, FPS: {fps}')
    return fps_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config yaml', default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)
    gpu_benchmark = GPUBenchmark(config['settings'])

    for model in config['models']:
        print()
        print(model)
        batch_size = 1
        gpu_benchmark.run_test(config['models'][model], batch_size)
        #fps_batch = 
        #pprint.pprint(fps_batch)


if __name__ == '__main__':
    main()