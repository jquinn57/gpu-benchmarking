# source ~/onnx-env/bin/activate
import argparse
import glob
import os
from sbi4onnx import initialize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Input dir')
    parser.add_argument('--output_dir', help='Output dir')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_list = glob.glob(os.path.join(args.input_dir, '*.onnx'))
    for model_name in model_list:
        output_name = os.path.join(args.output_dir, os.path.basename(model_name))
        onnx_graph = initialize(input_onnx_file_path=model_name,output_onnx_file_path=output_name,initialization_character_string='batch')


if __name__ == "__main__":
    main()
