# source ~/onnx-env/bin/activate
import argparse
import pprint
import onnxruntime
import yaml
import time
import math
import numpy as np
import os
import pprint
import threading
import queue
from pmd_reader import PMDReader
from google_sheet_api import GoogleSheetAPI
from kasa_reader import KasaReader


class AutoGPUBenchmark:
    def __init__(self, settings):
        self.settings = settings
        self.pmd_reader = None
        self.kasa_reader = None
        
        self.running = False
        if 'pmd' in settings:
            self.pmd_reader = PMDReader(*settings['pmd'])
            if not self.pmd_reader.check_device():
                print('PMD not found')
                self.pmd_reader = None
        if 'kasa' in settings:
            self.kasa_reader = KasaReader(*settings['kasa'])


    def shutdown(self):
        if self.pmd_reader:
            self.pmd_reader.stop_reading()
        if self.kasa_reader:
            self.kasa_reader.stop_reading()

    def batch_worker(self, batch_q, res, batch_size, nbatches):

        print("starting image_worker thread")
        for n in range(nbatches):
            img_batch = np.random.random((batch_size, 3, res, res)).astype(np.float32)
            batch_q.put(img_batch)

    def run_test(self, onnx_filename, batch_size=1):
            """
            Run a test with one condition
            """

            self.num_images = self.settings['num_images']
            use_cuda = (self.settings['onnx_ep'].lower() == 'cuda')
            trt_options = { 'trt_engine_cache_enable': False, 
                            'trt_engine_cache_path': './trt_cache', 
                            'trt_fp16_enable': False,
                            'trt_int8_enable': False,
                            'trt_dla_enable': False,
                            'trt_int8_use_native_calibration_table': True, 
                            'trt_int8_calibration_table_name': onnx_filename.replace('onnx_model_0.onnx', 'calib.cache')}
            if use_cuda:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = [('TensorrtExecutionProvider', trt_options), 'CUDAExecutionProvider', 'CPUExecutionProvider']
            
            sess_options = onnxruntime.SessionOptions()

            session = onnxruntime.InferenceSession(onnx_filename, providers=providers, sess_options=sess_options)
            output_names = [x.name for x in session.get_outputs()]
            print(output_names)
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            print(input_name)
            print(input_shape)
            
            # dynamic resolution (assuming square)
            if input_shape[-1] == 'width':
                res = self.settings['resolution']
            else:
                res = input_shape[-1]

            # dynamic batch size
            if input_shape[0] == 'batch':
                batch_size = batch_size
            else:
                batch_size = input_shape[0]

            is_warm = False
            # add one batch for warm up
            nbatches = 1 + int(math.floor(self.num_images / batch_size))
            count = 0

            if self.pmd_reader:
                self.pmd_reader.start_reading()
            if self.kasa_reader:
                self.kasa_reader.start_reading()

            # option 1: create input once outside of loop (zeros or random)
            # img_batch = np.zeros((batch_size, 3, res, res)).astype(np.float32)

            batch_q = queue.Queue(maxsize=8)
            self.thread = threading.Thread(target=self.batch_worker, args=(batch_q, res, batch_size, nbatches))
            self.thread.start()

            t0 = time.perf_counter()
            for n in range(nbatches):

                # option 2: create new input each time through the loop (zeros or random)
                #img_batch = np.random.random((batch_size, 3, res, res)).astype(np.float32)
                #img_batch = np.zeros((batch_size, 3, res, res)).astype(np.float32)

                # option 3: create inputs in seperate thread, pull from queue
                img_batch = batch_q.get()

                y = session.run(output_names, {input_name: img_batch})[0]
                count += batch_size
                # restart the counter and timer after warmup batch
                if not is_warm:
                    count = 0
                    is_warm = True
                    t0 = time.perf_counter()
                    # clear out the power measurement queue
                    if self.pmd_reader:
                        self.pmd_reader.avg_recent_readings()

            dt = time.perf_counter() - t0
            time_per_image = 1000 * dt / count
            fps = int(round(count / dt))

            if self.pmd_reader:
                power_avg_pmd = self.pmd_reader.avg_recent_readings()
            else:
                power_avg_pmd = 0

            if self.kasa_reader:
                power_avg_kasa = self.kasa_reader.avg_recent_readings()
            else:
                power_avg_kasa = 0

            # worst case latency - time to wait to gather batch_size images plus inference time for batch
            latency_ms = (2 * batch_size - 1) * time_per_image

            output = {}
            output["batch_size"] = batch_size
            output["inference_time_ms"] = time_per_image
            output["fps"] = fps
            output["pcie_power"] = power_avg_pmd
            output["sys_power"] = power_avg_kasa
            output["count"] = count
            output["total_time_s"] = dt
            output["latency_ms"] = latency_ms
            output["resolution"] = res
            pprint.pprint(output)

            # wait for worker thread to finish before moving on
            self.thread.join()
            return output

def get_model_list(root_dir):
    # select a subset for Sam
    from itertools import product
    resolutions = [160, 224, 320, 480]
    versions = ['yolo3', 'yolo5', 'yolo8']
    sizes = ['n', 's', 'm']
    selected_models = [f'{v}{s}_{r}' for v, s, r in product(versions, sizes, resolutions)]

    model_list = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'onnx_dynamic.onnx' in filenames:
            model_name = os.path.basename(dirpath)
            if model_name not in selected_models:
                continue
            #model_path = os.path.join(dirpath, 'onnx_dynamic.onnx')
            model_path = os.path.join(dirpath, 'onnx_model_0.onnx')
            model_list.append((model_name, model_path))
    model_list.sort()
    print(model_list)
    print(f'Number of models: {len(model_list)}')
    return model_list

def convert_to_fp16(model_list):
    import onnx
    from onnxconverter_common import float16
    new_model_list = []
    print('converting models to FP16')
    for model_name, model_path in model_list:
        model_fp32 = onnx.load(model_path)
        model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
        new_model_path = model_path.replace('model_0.onnx', 'model_0_fp16.onnx')
        new_model_list.append((model_name, new_model_path))
        onnx.save(model_fp16, new_model_path)
    print(new_model_list)
    return new_model_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config yaml', default='gpu_auto_bench.yaml')
    parser.add_argument('--start_with', help='Model in sorted list to start with', default=None)
    parser.add_argument('--fp16', action='store_true', help='convert onnx model to FP16')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)
    
    onnx_model_path = config['settings']['model_path']
    model_list = get_model_list(onnx_model_path)
    if args.fp16:
        model_list = convert_to_fp16(model_list)

    bench = AutoGPUBenchmark(config['settings'])
    header = ['Model', 'Resolution', 'Batch Size', 'FPS', 'Latency(ms)', 'PCIe Power(W)', 'Sys Power(W)']
    gsapi = GoogleSheetAPI(config['settings']['google_sheet_name'])
    gsapi.open_worksheet(config['settings']['google_sheet_tab'])
    gsapi.append_row(header)

    skip_models = args.start_with is not None
    first_time = False
    batch_sizes = config['settings']['batch_sizes']
    for model_name, model_path in model_list:
        print('\n')
        print(model_name)
        print(model_path)
        if skip_models:
            if model_name != args.start_with:
                continue
            skip_models = False

        # let the first time through be a warmup
        if first_time:
            results = bench.run_test(model_path, batch_size=1)
            first_time = False

        for batch_size in batch_sizes:
            print(f'batch_size = {batch_size}')
            try:
                results = bench.run_test(model_path, batch_size=batch_size)
                row = [model_name, 
                    results['resolution'],
                    results['batch_size'],
                    results['fps'],
                    results['latency_ms'],
                    results['pcie_power'],
                    results['sys_power']]
                gsapi.append_row(row)
            except Exception as e:
                print(e)
                print('Error, skipping to next model')
                row = [model_name, '', batch_size, '', '', '', str(e)]
                gsapi.append_row(row)

    bench.shutdown()


if __name__ == "__main__":
    main()
