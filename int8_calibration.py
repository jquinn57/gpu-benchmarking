#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This script demonstrates how to use the Calibrator API provided by Polygraphy
to calibrate a TensorRT engine to run in INT8 precision.
"""

# export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.10/dist-packages
import numpy as np
import os
from polygraphy.backend.trt import (
    Calibrator,
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
)
from polygraphy.logger import G_LOGGER

# The data loader argument to `Calibrator` can be any iterable or generator that yields `feed_dict`s.
# A `feed_dict` is just a mapping of input names to corresponding inputs.
def calib_data(resolution):
    for _ in range(4):
        # TIP: If your calibration data is already on the GPU, you can instead provide GPU pointers
        # (as `int`s), Polygraphy `DeviceView`s, or PyTorch tensors instead of NumPy arrays.
        #
        # For details on `DeviceView`, see `polygraphy/cuda/cuda.py`.
        yield {"images": np.random.random(shape=(1, 3, resolution, resolution), dtype=np.float32)}  


def main(model_path):

    calib_path = model_path.replace('onnx_model_0.onnx', 'calib.cache')
    # extract resolution - hack specific to yolobench naming convention
    res = int(os.path.dirname(model_path).split('_')[-1])
    print(model_path)
    print(res)


    # We can provide a path or file-like object if we want to cache calibration data.
    # This lets us avoid running calibration the next time we build the engine.
    #
    # TIP: You can use this calibrator with TensorRT APIs directly (e.g. config.int8_calibrator).
    # You don't have to use it with Polygraphy loaders if you don't want to.
    calibrator = Calibrator(data_loader=calib_data(res), cache=calib_path)

    # We must enable int8 mode in addition to providing the calibrator.
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(model_path),
        config=CreateConfig(int8=True, calibrator=calibrator),
    )

    # When we activate our runner, it will calibrate and build the engine. If we want to
    # see the logging output from TensorRT, we can temporarily increase logging verbosity:
    with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(build_engine) as runner:
        # Finally, we can test out our int8 TensorRT engine with some dummy input data:
        inp_data = np.random.random(shape=(1, 3, res, res), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer({"images": inp_data})

        # assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!


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


if __name__ == "__main__":
    model_list = get_model_list('/ssd/yolobench')
    for model_name, model_path in model_list:
        main(model_path)