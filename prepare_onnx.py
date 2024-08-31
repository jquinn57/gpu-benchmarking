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
        print('\n')
        print(output_name)
        # if disable_onnxsim is False then onnx simplify is run and can cause some problems
        # In the yolov5 models there are Resize nodes which use the sizes input which includes batch size which gets simplfied down to a constant tensor
        # before the batch size is changed to 'batch'
        onnx_graph = initialize(input_onnx_file_path=model_name,output_onnx_file_path=output_name, initialization_character_string='batch', disable_onnxsim=True)


if __name__ == "__main__":
    main()
