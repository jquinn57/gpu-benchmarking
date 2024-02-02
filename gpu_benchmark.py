import yaml
import argparse
import pprint
import onnxruntime
import time
import numpy as np
import cv2
import glob
import math

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
    
    for batch_size in model_config['batch_sizes']:
        nbatches = int(math.floor(settings['num_images'] / batch_size))

        img_batch = np.random.random((batch_size, 3, res, res)).astype(np.float32)
        # warm up batch
        y = session.run(output_names, {input_name: img_batch})[0]
        count = 0

        t0 = time.perf_counter()
        for n in range(nbatches):
            y = session.run(output_names, {input_name: img_batch})[0]
            count += batch_size

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

    for model in config['models']:
        print()
        print(model)
        fps_batch = run_test(config['settings'], config['models'][model])
        pprint.pprint(fps_batch)


if __name__ == '__main__':
    main()