import onnxruntime
import os
from IPython import embed
import glob
import cv2
import numpy as np
import time
import timeit


class MyData:
    def __init__(self, id):
        self.__id = id

    def get_id(self):
        return self.__id

def callback(res: np.ndarray, data: MyData, err: str) -> None:
    print('output')
    print(len(res))
    print(res[0].shape)

res = 640
image_list = glob.glob('/home/jquinn/datasets/coco/images/val2017/*.jpg')
img = cv2.imread(image_list[0])
img = cv2.resize(img, (res, res),interpolation=cv2.INTER_LINEAR).astype(np.float32)  / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, 0)
print(img.shape)


onnx_filename = './dfp_models/model_0_yolov5n-SiLU-640.onnx'
processor_onnx = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

fmaps = processor_onnx.run(None, {'images': img})

onnx_post_filename = onnx_filename.replace('.onnx', '_post.onnx')

options = onnxruntime.SessionOptions()
options.enable_profiling=False
#options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL

post_processor_onnx = onnxruntime.InferenceSession(onnx_post_filename, sess_options=options, providers=['CPUExecutionProvider'])

input_names = [c.name for c in post_processor_onnx.get_inputs()]
input_shapes = [c.shape for c in post_processor_onnx.get_inputs()]

fmap_dict = {}
for name, fmap in zip(input_names, fmaps):
    fmap_dict[name] = fmap

my_data = MyData(123456)


t0 = time.perf_counter()
post_processor_onnx.run_async(None, fmap_dict, callback, my_data)
dt = (time.perf_counter() - t0)

print(1000*dt)
print(1/dt)

time.sleep(3)

print('done')