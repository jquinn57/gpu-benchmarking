# source ~/onnx-env/bin/activate
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
import pprint
import queue
from pmd_reader import PMDReader
from kasa_reader import KasaReader


class GPUBenchmark:
    def __init__(self, settings):
        self.settings = settings
        self.pmd_reader = None
        if "pmd" in settings:
            self.pmd_reader = PMDReader(*settings["pmd"])
            if not self.pmd_reader.check_device():
                print("PMD not found")
                self.pmd_reader = None
        self.kasa_reader = None
        if "kasa" in settings:
            self.kasa_reader = KasaReader(*settings["kasa"])

    def image_worker(self, res):
        # empty out the queue if there were some leftovers from previous run
        # which don't fit evenly into a batch
        while not self.image_q.empty():
            img = self.image_q.get()
        print("starting image_worker thread")
        print(f"resolution: {res}")
        n = 0
        while self.running and n < self.num_images:
            img_filename = self.image_list[n]
            img = cv2.imread(img_filename)
            img = cv2.resize(img, (res, res)).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            self.image_q.put(img)
            n += 1

    def run_test(self, model_config, batch_size, do_inference=True):
        """
        Run a test with one condition
        """

        if "image_path" in self.settings and self.settings["image_path"]:
            self.load_images = True
            self.image_list = glob.glob(self.settings["image_path"] + "/*.jpg")
            self.num_images = min(len(self.image_list), self.settings["num_images"])
            self.image_q = queue.Queue()
            self.running = False
        else:  # use random data instead of actual images
            self.load_images = False
            self.image_list = []
            self.num_images = self.settings["num_images"]

        if do_inference:
            use_cuda = self.settings["onnx_ep"] == "cuda"
            providers = (
                ["CUDAExecutionProvider"] if use_cuda else ["TensorrtExecutionProvider"]
            )
            session = onnxruntime.InferenceSession(
                model_config["filename"], providers=providers
            )
            output_names = [x.name for x in session.get_outputs()]
            print(output_names)
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            print(input_name)
            print(input_shape)

        res = model_config["resolution"]

        is_warm = False
        nbatches = int(math.floor(self.num_images / batch_size))
        count = 0

        if self.load_images:
            self.thread = threading.Thread(
                target=self.image_worker, args=(model_config["resolution"],)
            )
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
                    img_new = self.image_q.get()
                    img_batch[i, :, :, :] = img_new
            else:
                img_batch = np.random.random((batch_size, 3, res, res)).astype(
                    np.float32
                )

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
                # clear out the power measurement queue
                if self.kasa_reader:
                    self.kasa_reader.avg_recent_readings()
                if self.pmd_reader:
                    self.pmd_reader.avg_recent_readings()

        dt = time.perf_counter() - t0
        time_per_image = 1000 * dt / count
        fps = int(round(count / dt))

        if self.pmd_reader:
            power_avg_pmd = int(round(self.pmd_reader.avg_recent_readings()))
        else:
            power_avg_pmd = 0

        if self.kasa_reader:
            power_avg_kasa = int(round(self.kasa_reader.avg_recent_readings()))
        else:
            power_avg_kasa = 0

        # worst case latency - time to wait to gather batch_size images plus inference time for batch
        latency_ms = (2 * batch_size - 1) * time_per_image

        output = {}
        output["batch_size"] = batch_size
        output["inference_time_ms"] = time_per_image
        output["fps"] = fps
        output["system_power"] = power_avg_kasa
        output["pcie_power"] = power_avg_pmd
        output["count"] = count
        output["total_time_s"] = dt
        output["latency_ms"] = latency_ms
        pprint.pprint(output)

        # wait for worker thread to finish before moving on
        self.running = False
        self.thread.join()

        return output

    def shutdown(self):
        if self.kasa_reader:
            self.kasa_reader.stop_reading()
        if self.pmd_reader:
            self.pmd_reader.stop_reading()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Path to config yaml", default="config_gpu.yaml"
    )
    parser.add_argument(
        "--output_csv", help="Path to output csv", default="output_gpu.csv"
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)
    gpu_benchmark = GPUBenchmark(config["settings"])

    outputs = [
        "Model, Resolution, Batch Size, FPS, Latency(ms), PCIe Power, System Power"
    ]
    for model in config["models"]:
        print()
        print(model)
        model_config = config["models"][model]
        res = model_config["resolution"]

        # data = gpu_benchmark.run_test(model_config, 1, do_inference=False)
        # outputs.append(f"{model}, {res}, {0}, {0}, {0}, {data['pcie_power']}, {data['system_power']}")

        for batch_size in model_config["batch_sizes"]:
            data = gpu_benchmark.run_test(model_config, batch_size, do_inference=True)
            outputs.append(
                f"{model}, {res}, {batch_size}, {data['fps']}, {data['latency_ms']}, {data['pcie_power']}, {data['system_power']}"
            )
        outputs.append("")

    gpu_benchmark.shutdown()

    with open(args.output_csv, "wt") as fp:
        for line in outputs:
            fp.write(line + "\n")


if __name__ == "__main__":
    main()
