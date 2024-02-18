# GPU Benchmarking

## CUDA, TensorRT, and ONNX Setup

Depending on which version of CUDA you have installed you may be able to simply do `pip install onnxruntime-gpu`.  Howeever, I had to compile the ONNX runtime from source to get it to work with CUDA 12.2

Building onnxrutime from source (with support for CUDA execution provider):

https://onnxruntime.ai/docs/build/eps.html#cuda

```
source ~/onnx-env/bin/activate
sudo apt install python3-numpy
sudo apt install python3-dev
pip install pybind11 wheel pytest onnx
./build.sh --use_cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --cuda_home /usr/local/cuda-12/  --config Release --build_shared_lib --parallel --build_wheel --enable_pybind
pip install /home/jquinn/onnxruntime/build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl
```

To enable the TensorRT execution provider, you also have to install TensorRT and also include additional arguments to the build:
```
--use_tensorrt --tensorrt_home /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0
```

The versions that I used were:
```
CUDA 12.2 (driver 535.129)
CUDA Toolkit 12.2
cuDNN 8.9.7.29-1+cuda12.2
TensorRT 8.6.1.6-1+cuda12.0
ONNX runtime 1.17.0 (compiled from source)
```

## Power measurement hardware

### Power measuring device

This device will measure the current which goes through the PCIe power cable which is plugged into the GPU.

Documentation: https://www.elmorlabs.com/product/elmorlabs-pmd-usb-power-measurement-device-with-usb/

To enable reading over USB, install the CH341 driver from here: https://github.com/WCHSoftGroup/ch341ser_linux
and follow instructions to build and install.

(If device is not showing up in lsusb check dmesg. I had a conflict with brltty which was being loaded because it was listed in a udev rules with the same vendor and product ID. I removed that rule.)

### Kasa Smart Power strip

https://www.kasasmart.com/us/products/smart-plugs/kasa-smart-wi-fi-power-strip-hs300

This device has 6 outlets and can measure power used by each one.  It will connect to your WiFi network.  The data can be viewed in an Android app, and can be queried via a Python API: https://github.com/python-kasa/python-kasa

`pip install python-kasa`




## Configuration
