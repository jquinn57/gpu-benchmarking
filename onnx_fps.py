import onnxruntime
import numpy as np
import cv2
import glob
import time
import argparse
from gpu_monitor import Monitor
from IPython import embed
import matplotlib.pyplot as plt
import seaborn as sns


def run_test(use_tensorrt, batch_size, onnx_file, res, nbatches=1000):

    providers = ['TensorrtExecutionProvider'] if use_tensorrt else ['CUDAExecutionProvider']
    print(providers)
    session = onnxruntime.InferenceSession(onnx_file, providers=providers)
    output_names = [x.name for x in session.get_outputs()]
    print(output_names)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(input_name)
    print(input_shape)

    count = 0
    img_batch = np.random.random((batch_size, 3, res, res)).astype(np.float32)
    # warm up batch
    y = session.run(output_names, {input_name: img_batch})[0]

    t0 = time.perf_counter()
    for n in range(nbatches):
        y = session.run(output_names, {input_name: img_batch})[0]
        count += batch_size

    dt = time.perf_counter() - t0
    time_per_image = 1000 * dt / count
    fps = count / dt
    print(f'Time per image: {time_per_image} ms, FPS: {fps}')
    return fps

def run(onnx_filename, batch_sizes, res):

    stats = np.zeros((len(batch_sizes), 4))

    mon = Monitor()
    for i, bs in enumerate(batch_sizes):
        fps = run_test(False, bs, onnx_filename, res)
        mon_stats = mon.get_vals().mean(axis=0)
        print(mon_stats)
        stats[i, 0] = bs
        stats[i, 1] = fps
        stats[i, 2] = mon_stats[1]  # utilization
        stats[i, 3] = mon_stats[2]  # mem
        time.sleep(1)

    mon.stop()
    print(stats)
    np.save('stats.npy', stats)

def plot_figure(filename='stats.npy', title=None, plot_filename=None):
    stats = np.load(filename)
    sns.set_theme()

    fig, ax1 = plt.subplots()

    ax1.plot(stats[:, 0], stats[:, 1], color='g', marker='o', label='FPS')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('FPS', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    ax2.plot(stats[:, 0], stats[:, 3], color='b', marker='o',label='GPU memory used')
    ax2.set_ylabel('Memory %', color='b')
    ax2.set_ylim([0, 100])
    ax2.tick_params(axis='y', labelcolor='b')

    if title is not None:
        plt.title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.grid(False)
    if plot_filename is not None:
        plt.savefig(plot_filename)
    plt.show()


if __name__ == '__main__':

    resnet = True
    if resnet:
        batch_sizes = [1, 2, 4, 8, 16, 32,  64, 128, 256]
        res = 224
        plot_filename = 'resnet152_fps_vs_batch_size.png'
        title = 'Resnet152 on RTX 5000'
        onnx_filename = 'resnet152.onnx'
    else:
        # yolov5
        batch_sizes = [1, 2, 3, 4, 6]
        res = 640
        plot_filename = 'yolov5s_fps_vs_batch_size.png'
        title = 'Yolov5s on RTX 4060'
        onnx_filename = 'yolov5s-leakyrelu3-dynamic.onnx'

    run(onnx_filename, batch_sizes, res)

    plot_figure(plot_filename=plot_filename, title=title)
