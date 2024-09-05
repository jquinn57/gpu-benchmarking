# source ~/onnx-env/bin/activate
import argparse
import os
from sbi4onnx import initialize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', help='Input root dir')
    args = parser.parse_args()

    count = 0
    for dirpath, dirnames, filenames in os.walk(args.input_root):
        if 'onnx_model_0.onnx' in filenames:
            model_name = os.path.basename(dirpath)
            print()
            print(count)
            print(model_name)
            input_path = os.path.join(dirpath, 'onnx_model_0.onnx' )
            output_path = os.path.join(dirpath, 'onnx_dynamic.onnx')
            print(input_path)
            print(output_path)
            count += 1
            # if disable_onnxsim is False then onnx simplify is run and can cause some problems
            # In the yolov5 models there are Resize nodes which use the sizes input which includes batch size which gets simplfied down to a constant tensor
            # before the batch size is changed to 'batch'
            onnx_graph = initialize(input_onnx_file_path=input_path,output_onnx_file_path=output_path, initialization_character_string='batch', disable_onnxsim=True)

if __name__ == "__main__":
    main()
