import os
import subprocess
import argparse
import yaml
import pprint
from google_sheet_api import GoogleSheetAPI
from kasa_reader import KasaReader

def get_model_list(root_dir):
    # select a subset for Sam
    from itertools import product
    resolutions = [160, 224, 320, 480]
    versions = ['yolo3', 'yolo5', 'yolo8']
    sizes = ['n', 's', 'm']
    selected_models = [f'{v}{s}_{r}' for v, s, r in product(versions, sizes, resolutions)]

    model_list = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'onnx_model_0.onnx' in filenames:
            model_name = os.path.basename(dirpath)
            if model_name not in selected_models:
                continue
            model_path = os.path.join(dirpath, 'onnx_model_0.onnx')
            model_list.append((model_name, model_path))
    model_list.sort()
    print(model_list)
    print(f'Number of models: {len(model_list)}')
    return model_list


def get_throughput(onnx_path, kasa_reader):

    engine_path = onnx_path.replace('.onnx', '.engine')
    command = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={onnx_path}",
        f"--int8",
        f"--loadEngine={engine_path}"
    ]
    
    kasa_reader.start_reading()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    throughput_line = None
    for line in iter(process.stdout.readline, ''):
        print(line, end="")
        if "Throughput:" in line:
            throughput_line = line.strip()
    
    process.wait()
    power_avg_kasa = kasa_reader.avg_recent_readings()

    fps = float(throughput_line.split(' ')[-2])
    return fps, power_avg_kasa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config yaml', default='run_trtexec.yaml')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)
    
    onnx_model_path = config['settings']['model_path']

    model_list = get_model_list(onnx_model_path)

    header = ['Model', 'FPS', 'Sys Power (W)']
    gsapi = GoogleSheetAPI(config['settings']['google_sheet_name'])
    gsapi.open_worksheet(config['settings']['google_sheet_tab'])
    gsapi.append_row(header)

    kasa_reader = KasaReader(*config['settings']['kasa'])

    for model_name, model_path in model_list:

        print('\n')
        print(model_name)
        print(model_path)
        fps, power_avg_kasa = get_throughput(model_path, kasa_reader)
        row = [model_name, fps, power_avg_kasa]
        print(fps)
        gsapi.append_row(row)


    kasa_reader.stop_reading()



if __name__ == '__main__':
    main()
