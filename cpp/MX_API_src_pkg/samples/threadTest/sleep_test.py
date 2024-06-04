import matplotlib.pyplot as plt
import numpy as np
import subprocess
from IPython import embed

input_sleep_us = [0, 100, 200, 300, 400, 800]
output_sleep_us = [0, 5, 10, 20, 50, 100, 200, 400, 800]
working_dir = '/home/jquinn/gpu-benchmarking/cpp/MX_API_src_pkg/build/samples/threadTest'
data = np.zeros( (len(input_sleep_us), len(output_sleep_us)), dtype=float)

outputs_all = ['input_sleep_us, output_sleep_us, fps']

run_test = True
if run_test:
    for i, in_sleep in enumerate(input_sleep_us):
        for j, out_sleep in enumerate(output_sleep_us):
            cmd = f'./threadTest {in_sleep} {out_sleep}'
            print(cmd)
            result = subprocess.run(cmd.split(), capture_output=True, text=True, cwd=working_dir)
            outputs = result.stdout.split(',')
            fps = float(outputs[2])
            print(fps)
            data[i, j] = fps
            outputs_all.append(f'{in_sleep}, {out_sleep}, {fps}')

    print(data)

    with open('results2.csv', 'wt') as fp:
        for line in outputs_all:
            fp.write(line + '\n')
    
    np.save('results2.npy', data)

else:

    data = np.load('results2.npy')
    data = data[1:, :]
    input_sleep_us = input_sleep_us[1:]
    num_rows = data.shape[0]

    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 2 * num_rows), sharex=True, sharey=True)

    for i, in_sleep in enumerate(input_sleep_us):
        axes[i].plot(output_sleep_us, data[i, :], 'o-', label=f'Input sleep us = {in_sleep}')
        axes[i].grid(True)
        axes[i].set_ylim([0, 500])
        axes[i].legend()


    fig.text(0.5, 0.04, 'Output sleep us', ha='center')
    fig.text(0.04, 0.5, 'FPS', va='center', rotation='vertical')

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.savefig('sleep-test-results2.png')

    plt.show()


