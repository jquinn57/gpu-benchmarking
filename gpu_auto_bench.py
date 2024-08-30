# source ~/onnx-env/bin/activate
import argparse
import pprint
import onnxruntime
import yaml
import time
import math
import numpy as np
import glob
import os
import pprint
import threading
import queue
from pmd_reader import PMDReader
from google_sheet_api import GoogleSheetAPI


class AutoGPUBenchmark:
    def __init__(self, settings):
        self.settings = settings
        self.pmd_reader = None
        self.batch_q = queue.Queue()
        self.running = False
        if 'pmd' in settings:
            self.pmd_reader = PMDReader(*settings['pmd'])
            if not self.pmd_reader.check_device():
                print('PMD not found')
                self.pmd_reader = None

    def shutdown(self):
        if self.pmd_reader:
            self.pmd_reader.stop_reading()

    def batch_worker(self, res, batch_size, nbatches):
        # empty out the queue if there were some leftovers from previous run (shoudlnt happen)
        while not self.batch_q.empty():
            img = self.batch_q.get()
            print('leftover')
        print("starting image_worker thread")
        for n in range(nbatches):
            img_batch = np.random.random((batch_size, 3, res, res)).astype(np.float32)
            self.batch_q.put(img_batch)

    def run_test(self, onnx_filename, batch_size=1):
            """
            Run a test with one condition
            """

            self.num_images = self.settings['num_images']
            use_cuda = (self.settings['onnx_ep'].lower() == 'cuda')
            trt_options = { 'trt_engine_cache_enable': True, 'trt_engine_cache_path': './trt_cache'}
            if use_cuda:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = [('TensorrtExecutionProvider', trt_options), 'CUDAExecutionProvider', 'CPUExecutionProvider']
            
            session = onnxruntime.InferenceSession(onnx_filename, providers=providers)
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

            # option 1: create input once outside of loop (zeros or random)
            # img_batch = np.zeros((batch_size, 3, res, res)).astype(np.float32)

            self.thread = threading.Thread(target=self.batch_worker, args=(res, batch_size, nbatches))
            self.thread.start()

            t0 = time.perf_counter()
            for n in range(nbatches):

                # option 1: create new input each time through the loop (zeros or random)
                #img_batch = np.random.random((batch_size, 3, res, res)).astype(np.float32)
                #img_batch = np.zeros((batch_size, 3, res, res)).astype(np.float32)

                # option 3: create inputs in seperate thread, pull from queue
                img_batch = self.batch_q.get()

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

            # worst case latency - time to wait to gather batch_size images plus inference time for batch
            latency_ms = (2 * batch_size - 1) * time_per_image

            output = {}
            output["batch_size"] = batch_size
            output["inference_time_ms"] = time_per_image
            output["fps"] = fps
            output["pcie_power"] = power_avg_pmd
            output["count"] = count
            output["total_time_s"] = dt
            output["latency_ms"] = latency_ms
            output["resolution"] = res
            pprint.pprint(output)

            # wait for worker thread to finish before moving on
            self.thread.join()
            return output


# def inspect_shapes(model):
#     graph = model.graph
#     # Inspect input sizes
#     print("Inputs:")
#     for input_tensor in graph.input:
#         input_name = input_tensor.name
#         input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
#         print(f"{input_name}: {input_shape}")

#     # Inspect output sizes
#     print("Outputs:")
#     for output_tensor in graph.output:
#         output_name = output_tensor.name
#         output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
#         print(f"{output_name}: {output_shape}")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help='Path to config yaml', default='gpu_auto_bench.yaml')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)
    
    onnx_model_path = config['settings']['model_path']
    model_list = glob.glob(os.path.join(onnx_model_path, '*.onnx'))
    model_list.sort()
    pprint.pprint(model_list)
    num_models = len(model_list)
    print(num_models)

    bench = AutoGPUBenchmark(config['settings'])
    header = ['Model', 'Resolution', 'Batch Size', 'FPS', 'Latency(ms)', 'PCIe Power(W)']
    gsapi = GoogleSheetAPI('GPU-Auto-Bench1')
    gsapi.open_worksheet('0')
    gsapi.append_row(header)


    for model_name in model_list:
        print('\n')
        results = bench.run_test(model_name)
        model_name_short = os.path.basename(model_name).replace('.onnx', '')
        row = [model_name_short, 
               results['resolution'],
               results['batch_size'],
               results['fps'],
               results['latency_ms'],
               results['pcie_power']]
        gsapi.append_row(row)

    bench.shutdown()


if __name__ == "__main__":
    main()
