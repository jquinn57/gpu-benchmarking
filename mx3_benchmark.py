# source ~/mx3-env/bin/activate
import yaml
import argparse
import pprint
import time
import numpy as np
import cv2
import glob
import os

# import multiprocessing
import threading
from queue import Queue
import onnxruntime
from kasa_reader import KasaReader
from memryx import AsyncAccl
from memryx import Benchmark
import logging

logging.basicConfig(level=logging.INFO)


class MX3Benchmark:
    def __init__(self, settings):
        self.settings = settings
        self.do_post_processing = settings.get("post_processing", False)
        self.img_queue = Queue()
        self.output_queue = Queue()
        self.post_processor_onnx = None
        self.input_names = None

        self.kasa_reader = None
        if "kasa" in settings:
            self.kasa_reader = KasaReader(*settings["kasa"])

        self.count = 0
        if "image_path" in self.settings and self.settings["image_path"]:
            self.load_images = True
            self.image_list = glob.glob(self.settings["image_path"] + "/*.jpg")
            self.num_images = min(len(self.image_list), self.settings["num_images"])
        else:  # use zeros instead of actual images
            self.load_images = False
            self.image_list = []
            self.num_images = self.settings["num_images"]

    @staticmethod
    def crop_or_pad(img, target_shape):
        pad_y = max(target_shape[0] - img.shape[0], 0)
        pad_x = max(target_shape[1] - img.shape[1], 0)
        img = np.pad(img, ((0, pad_y), (0, pad_x), (0, 0)))
        img = img[: target_shape[0], : target_shape[1], :]
        return img

    def image_preprocessor(self):

        print(f'starting image_preprocessing, img_queue size: {self.img_queue.qsize()}')
        dt_tot = 0
        if self.load_images:
            for img_filename in self.image_list[: self.num_images]:
                img = cv2.imread(img_filename)
                t0 = time.perf_counter()
                img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                #img = self.crop_or_pad(img, (self.resolution, self.resolution, 3)).astype(np.float32) / 255.0
                dt = time.perf_counter() - t0
                dt_tot += dt
                self.img_queue.put(img)
        else:
            for n in range(self.num_images):
                img = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)
                self.img_queue.put(img)
        dt_avg = dt_tot / self.num_images
        print(f"Preprocessor done, dt_avg = {dt_avg}")

    def postprocessor(self):
        count = 0

        while count < self.num_images:
            fmap_dict = self.output_queue.get()
            if self.post_processor_onnx is not None:
                output = self.post_processor_onnx.run(None, fmap_dict)[0]
            count += 1

        print("Postprocessor done")

    def data_source(self):
        count = 0
        while count < self.num_images:
            yield self.img_queue.get()
            count += 1

    def output_processor(self, *outputs):

        input_dict = {}
        for i, output in enumerate(outputs):
            name = self.input_names[i]
            input_dict[name] = np.moveaxis(output[None, :, :, :], -1, 1)
        self.output_queue.put(input_dict)
        # print(f'output: {post_output.shape}')
        self.count += 1

    def baseline_power_test(self, model_config):
        dfp_filename = model_config["filename"]
        self.resolution = model_config["resolution"]
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
        dfp_filename = model_config["filename"]
        self.resolution = model_config["resolution"]
        print('asyncaccl pre')
        accl = AsyncAccl(dfp_filename, chip_gen=3.1)
        print('asyncaccl post')

        self.input_names = ["0", "1", "2"]
        if self.do_post_processing:
            model_dir = os.path.dirname(dfp_filename)
            dfp_basename = os.path.basename(dfp_filename)
            onnx_filename = os.path.join(
                model_dir, "model_0_" + dfp_basename.replace(".dfp", "_post.onnx")
            )
            if os.path.exists(onnx_filename):
                print(f"Including post processing {onnx_filename}")
                self.post_processor_onnx = onnxruntime.InferenceSession(
                    onnx_filename, providers=["CPUExecutionProvider"]
                )
                self.input_names = [c.name for c in self.post_processor_onnx.get_inputs()]
            else:
                raise Exception(f"{onnx_filename} not found")

        self.count = 0
        print(f"Start test of {dfp_filename}")
        preprocessor = threading.Thread(target=self.image_preprocessor)
        t0_pre = time.perf_counter()
        preprocessor.start()

        #t1_pre = time.perf_counter()
        #dt_pre_ms = 1000*(t1_pre - t0_pre)
        #print(f'Pre-processing done, starting timer, time: {dt_pre_ms} ms')
        t0 = time.perf_counter()

        accl.connect_input(self.data_source)
        accl.connect_output(self.output_processor)

        postprocessor = threading.Thread(target=self.postprocessor)
        postprocessor.start()

        accl.wait()  # wait for the accelerator to finish execution
        print("Accl finished")
        postprocessor.join()
        preprocessor.join()

        t1 = time.perf_counter()
        # undocumented method?
        accl.stop()
        #accl.shutdown()
        print("Done")

        if self.kasa_reader:
            power_avg_kasa = int(round(self.kasa_reader.avg_recent_readings()))
        else:
            power_avg_kasa = 0

        dt = t1 - t0
        fps = self.count / dt
        inference_time_ms = 1000 * dt / self.count

        output = {}
        output["batch_size"] = 1
        output["inference_time_ms"] = inference_time_ms
        output["fps"] = int(round(fps))
        output["system_power"] = power_avg_kasa
        output["pcie_power"] = 0
        output["count"] = self.count
        output["total_time_s"] = dt
        output["latency_ms"] = 0
        pprint.pprint(output)
        return output

    def shutdown(self):
        if self.kasa_reader:
            self.kasa_reader.stop_reading()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config yaml", default="config_mx3.yaml")
    parser.add_argument("--output_csv", help="Path to output csv", default="output_mx3.csv")
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)
    mx3_benchmark = MX3Benchmark(config["settings"])

    outputs = ["Model, Resolution, Batch Size,  Latency(ms), FPS, FPS (mx_bench), System Power"]
    num_images = int(config['settings']['num_images'])

    for model in config["models"]:
        print()
        print(model)
        model_config = config["models"][model]
        res = model_config["resolution"]
        batch_size = 1

        # doing the Benchmark tests before the run_test call changes the results
        # Do a seperate latency measurement
        with Benchmark(dfp=model_config["filename"], verbose=2, chip_gen=3.1) as bm:
            _, latency_ms, _ = bm.run(threading=False)
        with Benchmark(dfp=model_config["filename"], verbose=2, chip_gen=3.1) as bm:
            _, _, fps_bm = bm.run(frames=num_images)
            fps_bm = int(round(fps_bm))

        time.sleep(1)
        data = mx3_benchmark.run_test(model_config)
        time.sleep(1)


        data["latency_ms"] = latency_ms
        outputs.append(
            f"{model}, {res}, {batch_size}, {data['latency_ms']}, {data['fps']},  {fps_bm}, {data['system_power']}"
        )

    mx3_benchmark.shutdown()
    with open(args.output_csv, "wt") as fp:
        for line in outputs:
            print(line)
            fp.write(line + "\n")


if __name__ == "__main__":
    main()
