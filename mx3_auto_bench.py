# source ~/mx/bin/activate
import argparse
import pprint
import yaml
import numpy as np
import os
import json
import pprint
from kasa_reader import KasaReader
from google_sheet_api import GoogleSheetAPI
from memryx import Benchmark
import time


class AutoMX3Benchmark:
    def __init__(self, settings):
        self.settings = settings
        self.kasa_reader = None
        
        self.running = False
        if 'kasa' in settings:
            self.kasa_reader = KasaReader(*settings['kasa'])

    def shutdown(self):
        if self.kasa_reader:
            self.kasa_reader.stop_reading()

    def run_test(self, dfp_filename):
            """
            Run a test with one condition
            """

            num_images = self.settings['num_images']
         
            if self.kasa_reader:
                self.kasa_reader.start_reading()

            with Benchmark(dfp=dfp_filename) as accl:
                _, _, fps = accl.run(frames=num_images)

            if self.kasa_reader:
                power_avg_kasa = self.kasa_reader.avg_recent_readings()
            else:
                power_avg_kasa = 0

            with Benchmark(dfp=dfp_filename) as accl:
                # single frame, get latency
                _, latency_ms, _ = accl.run(threading=False)

            # extract resolution - hack specific to yolobench naming convention
            res = int(dfp_filename.split('_')[-1].replace('/model.dfp', ''))
            output = {}
            output["batch_size"] = 1
            output["fps"] = fps
            output["sys_power"] = power_avg_kasa
            output["count"] = num_images
            output["latency_ms"] = latency_ms
            output["resolution"] = res
            pprint.pprint(output)
            return output

def get_model_list(root_dir):
    model_list = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'model.dfp' in filenames:
            model_name = os.path.basename(dirpath)
            model_path = os.path.join(dirpath, 'model.dfp')
            model_list.append((model_name, model_path))
    model_list.sort()
    return model_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config yaml', default='mx3_auto_bench.yaml')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)
    
    model_list = get_model_list(config['settings']['model_path'])

    # check local file for last model attemtped
    last_model_attempted = None
    if os.path.exists('last_model_attempted.txt'):
        with open('last_model_attempted.txt', 'rt') as fp:
            last_model_attempted = fp.readline().strip()
        
    bench = AutoMX3Benchmark(config['settings'])
    header = ['Model', 'Resolution', 'Batch Size', 'FPS', 'Latency(ms)', 'Sys Power(W)']
    gsapi = GoogleSheetAPI(config['settings']['google_sheet_name'])
    gsapi.open_worksheet(config['settings']['google_sheet_tab'])

    # check spreadsheet and start back on first unprocessed model
    data_mx3 = gsapi.get_dataframe('A1:F712', includes_header=True)
    if len(data_mx3) == 0:
        gsapi.append_row(header)
        last_processed = None
    else:
        last_processed = data_mx3.Model.iloc[-1]
    
    print(f'Last processed: {last_processed}')
    print(f'Last attempted: {last_model_attempted}')
    if last_model_attempted and last_processed and last_model_attempted != last_processed:
        # the last attempted model was not finished, so there was a failure. Record this to the spreadsheet and move on
        row = [last_model_attempted, '', '', '', '', '', 'Failure']
        gsapi.append_row(row)
        last_processed = last_model_attempted

    model_names = [mn[0] for mn in model_list]
    start_idx = 0 
    if last_processed:
        start_idx = model_names.index(last_processed) + 1

    first_time = False
    batch_size = 1

    for model_name, model_path in model_list[start_idx:]:
        print('\n')
        print(model_name)
        print(model_path)

        # let the first time through be a warmup
        if first_time:
            results = bench.run_test(model_path)
            first_time = False

        try:
            with open('last_model_attempted.txt', 'wt') as fp:
                fp.write(model_name)
            
            results = bench.run_test(model_path)
            row = [model_name, 
                results['resolution'],
                results['batch_size'],
                results['fps'],
                results['latency_ms'],
                results['sys_power']]
            gsapi.append_row(row)
            time.sleep(5)

        except Exception as e:
            print(e)
            print('Error, skipping to next model')
            row = [model_name, '', batch_size, '', '', '', str(e)]
            gsapi.append_row(row)
            time.sleep(5)

    bench.shutdown()


if __name__ == "__main__":
    main()
