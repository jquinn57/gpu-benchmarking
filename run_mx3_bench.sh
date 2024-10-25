#!/bin/bash
source /home/jquinn/mx/bin/activate
cd /home/jquinn/gpu-benchmarking
python mx3_auto_bench.py >> mx3.log 2>&1 

