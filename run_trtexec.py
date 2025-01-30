import os
import subprocess
from google_sheet_api import GoogleSheetAPI

def get_model_list(root_dir):
    # select a subset for Sam
    from itertools import product
    resolutions = [160, 224, 320, 480]
    versions = ['yolo3', 'yolo5', 'yolo8']
    sizes = ['n', 's', 'm']
    selected_models = [f'{v}{s}_{r}' for v, s, r in product(versions, sizes, resolutions)]

    model_list = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'onnx_model_0_fp16.onnx' in filenames:
            model_name = os.path.basename(dirpath)
            if model_name not in selected_models:
                continue
            model_path = os.path.join(dirpath, 'onnx_model_0_fp16.onnx')
            model_list.append((model_name, model_path))
    model_list.sort()
    print(model_list)
    print(f'Number of models: {len(model_list)}')
    return model_list


def get_throughput(onnx_path):

    engine_path = onnx_path.replace('.onnx', '.engine')
    command = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={onnx_path}",
        "--fp16",
        f"--saveEngine={engine_path}"
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    throughput_line = None
    for line in iter(process.stdout.readline, ''):
        print(line, end="")
        if "Throughput" in line:
            throughput_line = line.strip()
    
    process.wait()
    fps = float(throughput_line.split(' ')[-2])
    return fps


if __name__ == '__main__':
    model_list = get_model_list('/ssd/yolobench')

    header = ['Model', 'FPS']
    gsapi = GoogleSheetAPI('Jetson-Orin-Nano-Super+AGX32')
    gsapi.open_worksheet('375192165')
    gsapi.append_row(header)

    for model_name, model_path in model_list:

        print('\n')
        print(model_name)
        print(model_path)
        fps = get_throughput(model_path)
        row = [model_name, fps]
        print(fps)
        gsapi.append_row(row)
